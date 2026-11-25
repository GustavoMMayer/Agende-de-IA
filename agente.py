import json
import os
import re
from datetime import datetime, timedelta
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

# Configurações gerais e arquivos de persistência
console = Console()
MODELO_LLM = "qwen2.5:1.5b"
ARQUIVO_AGENDAMENTOS = "agendamentos.txt"
ARQUIVO_INACABADOS = "agendamentos_inacabados.txt"
ARQUIVO_LOG = "log_atendimentos.txt"


def gerar_protocolo() -> str:
    """
    Gera um identificador único de atendimento (ex: #AF-20261125-084412).
    """
    return f"#AF-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def registrar_log(protocolo: str, evento: str, detalhes: str) -> None:
    """
    Grava cronologicamente cada passo, tentativa e resposta no arquivo de log.
    """
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linha = f"[{timestamp}] [{protocolo}] [{evento}] {detalhes}\n"
    with open(ARQUIVO_LOG, "a", encoding="utf-8") as arquivo:
        arquivo.write(linha)


def salvar_agendamento_inacabado(
    protocolo: str,
    nome: str,
    telefone: str,
    veiculo: str,
    servico: str,
    data_tentada: str,
    horario_tentado: str,
    motivo: str,
) -> None:
    """
    Salva os dados parciais de clientes que informaram ao menos nome e telefone,
    mas não concluíram o agendamento, permitindo retorno ativo da equipe.
    """
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linha = (
        f"[{timestamp}] Protocolo: {protocolo} | Status: INACABADO ({motivo}) | "
        f"Cliente: {nome} | Telefone: {telefone} | Veículo: {veiculo} | "
        f"Serviço: {servico} | Última Data/Hora: {data_tentada} às {horario_tentado}\n"
    )
    with open(ARQUIVO_INACABADOS, "a", encoding="utf-8") as arquivo:
        arquivo.write(linha)
    registrar_log(protocolo, "INACABADO_SALVO", f"Motivo: {motivo} | Tel: {telefone}")


def obter_saudacao() -> str:
    """
    Retorna 'Bom dia', 'Boa tarde' ou 'Boa noite' conforme o horário do sistema.
    """
    hora_atual = datetime.now().hour
    if 5 <= hora_atual < 12:
        return "Bom dia"
    elif 12 <= hora_atual < 18:
        return "Boa tarde"
    else:
        return "Boa noite"


def validar_e_formatar_telefone(entrada_telefone: str) -> tuple[bool, str]:
    """
    Valida telefone brasileiro com ou sem DDD usando Regex e bloqueia sequências repetidas.
    Aceita:
      - Com DDD: (48) 99999-8888, 48999998888 (celular 11 dígitos) ou 4833334444 (fixo 10 dígitos)
      - Sem DDD: 99999-8888, 999998888 (celular 9 dígitos) ou 33334444 (fixo 8 dígitos)
    Retorna (True, telefone_formatado) ou (False, "").
    """
    apenas_digitos = re.sub(r"\D", "", entrada_telefone)

    # Remove prefixo internacional +55 caso o usuário digite
    if len(apenas_digitos) in (12, 13) and apenas_digitos.startswith("55"):
        apenas_digitos = apenas_digitos[2:]

    # Bloqueia números claramente falsos com todos os dígitos iguais (ex: 999999999)
    if len(apenas_digitos) >= 8 and len(set(apenas_digitos)) == 1:
        return False, ""

    # Regex completa: DDD opcional ([1-9]{2}), celular (9[6-9]\d{7}) ou fixo ([2-5]\d{7})
    padrao_telefone = re.compile(r"^(?:([1-9]{2}))?(?:(9[6-9]\d{7})|([2-5]\d{7}))$")
    match = padrao_telefone.match(apenas_digitos)

    if not match:
        return False, ""

    ddd, celular, fixo = match.groups()
    numero = celular if celular else fixo

    if len(numero) == 9:
        num_fmt = f"{numero[:5]}-{numero[5:]}"
    else:
        num_fmt = f"{numero[:4]}-{numero[4:]}"

    telefone_final = f"({ddd}) {num_fmt}" if ddd else num_fmt
    return True, telefone_final


def extrair_nome_telefone_com_ia(texto_usuario: str) -> dict:
    """
    Usa o LLM local para separar o nome do cliente e o telefone de contato
    a partir de uma resposta natural.
    """
    prompt_sistema = (
        "Você é o recepcionista da Oficina Mecânica Auto Fácil. "
        "Extraia o 'nome' do cliente e o 'telefone' da mensagem em JSON estrito.\n"
        "Se não encontrar algum deles, use 'NAO_INFORMADO'.\n"
        "Responda APENAS o JSON válido.\n\n"
        "Exemplos:\n"
        "Entrada: 'Meu nome é Carlos e meu zap é 48 99876-5432' -> {\"nome\": \"Carlos\", \"telefone\": \"48 99876-5432\"}\n"
        "Entrada: 'Ana, 991234567' -> {\"nome\": \"Ana\", \"telefone\": \"991234567\"}\n"
        "Entrada: 'Sou o Gustavo' -> {\"nome\": \"Gustavo\", \"telefone\": \"NAO_INFORMADO\"}"
    )

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": texto_usuario},
        ],
        options={"temperature": 0.0},
    )

    texto_bruto = resposta["message"]["content"].strip()
    dados = {"nome": "NAO_INFORMADO", "telefone": "NAO_INFORMADO"}
    match_json = re.search(r"\{.*\}", texto_bruto, re.DOTALL)

    if match_json:
        try:
            extraido = json.loads(match_json.group(0))
            dados.update(extraido)
        except json.JSONDecodeError:
            pass

    # Fallback por regex caso a IA não isole os dígitos do telefone na frase inicial
    if dados["telefone"] == "NAO_INFORMADO":
        match_tel = re.search(r"(\(?\d{2}\)?\s*\d{4,5}[-\s]?\d{4})", texto_usuario)
        if match_tel:
            dados["telefone"] = match_tel.group(1)

    return dados


def extrair_veiculo_servico_com_ia(texto_usuario: str) -> dict:
    """
    Usa o LLM local para extrair o modelo do veículo e o serviço/defeito desejado.
    """
    prompt_sistema = (
        "Você é o consultor técnico da Oficina Mecânica Auto Fácil. "
        "Extraia o 'veiculo' e o 'servico' (defeito ou manutenção desejada) em JSON estrito.\n"
        "Se não encontrar algum deles, use 'NAO_INFORMADO'.\n"
        "Responda APENAS o JSON válido.\n\n"
        "Exemplos:\n"
        "Entrada: 'Tenho um Onix 2020 e preciso trocar as pastilhas de freio' -> {\"veiculo\": \"Onix 2020\", \"servico\": \"trocar pastilhas de freio\"}\n"
        "Entrada: 'Fiat Palio, revisão geral' -> {\"veiculo\": \"Fiat Palio\", \"servico\": \"revisão geral\"}\n"
        "Entrada: 'Gol G5' -> {\"veiculo\": \"Gol G5\", \"servico\": \"NAO_INFORMADO\"}"
    )

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": texto_usuario},
        ],
        options={"temperature": 0.0},
    )

    texto_bruto = resposta["message"]["content"].strip()
    dados = {"veiculo": "NAO_INFORMADO", "servico": "NAO_INFORMADO"}
    match_json = re.search(r"\{.*\}", texto_bruto, re.DOTALL)

    if match_json:
        try:
            extraido = json.loads(match_json.group(0))
            dados.update(extraido)
        except json.JSONDecodeError:
            pass

    return dados


def extrair_data_horario_com_ia(mensagem_cliente: str) -> dict:
    """
    Usa o LLM local (RQ01) para extrair a data (incluindo dias da semana) e o horário.
    """
    prompt_sistema = (
        "Você é o assistente de agenda da Oficina Auto Fácil. "
        "Analise a mensagem e extraia 2 informações em JSON estrito:\n"
        "1. 'horario': hora no formato XXh (ex: '08h', '15h'). Se não houver, use 'NENHUM'.\n"
        "2. 'data': dia, data ou dia da semana citado (ex: 'Hoje', 'Amanhã', 'Sábado', 'Sexta', '28', '28/11'). "
        "Se não houver, use 'NAO_INFORMADA'.\n\n"
        "Responda APENAS o JSON válido.\n\n"
        "Exemplos:\n"
        "Entrada: 'Quero marcar às 14h' -> {\"horario\": \"14h\", \"data\": \"NAO_INFORMADA\"}\n"
        "Entrada: 'amanhã as 9 da manhã' -> {\"horario\": \"09h\", \"data\": \"Amanhã\"}\n"
        "Entrada: 'sábado às 10h' -> {\"horario\": \"10h\", \"data\": \"Sábado\"}\n"
        "Entrada: '28 as 15' -> {\"horario\": \"15h\", \"data\": \"28\"}"
    )

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": mensagem_cliente},
        ],
        options={"temperature": 0.0},
    )

    texto_bruto = resposta["message"]["content"].strip()
    dados = {"horario": "NENHUM", "data": "NAO_INFORMADA"}
    match_json = re.search(r"\{.*\}", texto_bruto, re.DOTALL)

    if match_json:
        try:
            extraido = json.loads(match_json.group(0))
            dados.update(extraido)
        except json.JSONDecodeError:
            pass

    # Normaliza o horário para 'XXh'
    busca_hora = re.search(r"\b(\d{1,2})h?\b", str(dados.get("horario", "")), re.IGNORECASE)
    if busca_hora and dados["horario"] != "NENHUM":
        dados["horario"] = busca_hora.group(1).zfill(2) + "h"
    else:
        dados["horario"] = "NENHUM"

    # Fallback determinístico para dias da semana ou dia numérico solto
    msg_lower = mensagem_cliente.lower()
    dias_semana_regex = r"\b(segunda|terca|terça|quarta|quinta|sexta|sabado|sábado|domingo)(?:-feira)?\b"
    match_semana = re.search(dias_semana_regex, msg_lower)

    if match_semana:
        dados["data"] = match_semana.group(1)
    elif str(dados.get("data", "")) == "NAO_INFORMADA":
        match_dia_solto = re.search(
            r"(?:^|\bdia\s+)(\d{1,2})(?:/(\d{1,2}))?(?=\s*(?:as|às|h|:|$))",
            msg_lower,
        )
        if match_dia_solto:
            dia = match_dia_solto.group(1)
            mes = match_dia_solto.group(2)
            dados["data"] = f"{dia}/{mes}" if mes else dia

    return dados


def resolver_data_segura(data_texto: str) -> tuple[bool, str, str]:
    """
    Converte a data informada para DD/MM/AAAA e valida regras de calendário:
    - Impede datas no passado (se o cliente pedir um dia já passado no mês, avança para o próximo mês).
    - Bloqueia agendamentos aos domingos (oficina fechada).
    Retorna (data_valida, data_formatada, mensagem_erro).
    """
    agora = datetime.now()
    hoje_data = agora.date()
    texto = data_texto.strip().lower().replace("-feira", "")

    data_alvo = None

    if texto in ["hoje", "", "nao_informada"]:
        data_alvo = hoje_data
    elif texto in ["amanhã", "amanha"]:
        data_alvo = hoje_data + timedelta(days=1)
    else:
        mapa_semana = {
            "segunda": 0,
            "terca": 1,
            "terça": 1,
            "quarta": 2,
            "quinta": 3,
            "sexta": 4,
            "sabado": 5,
            "sábado": 5,
            "domingo": 6,
        }

        if texto in mapa_semana:
            dia_semana_alvo = mapa_semana[texto]
            dias_a_frente = (dia_semana_alvo - hoje_data.weekday()) % 7
            if dias_a_frente == 0:
                dias_a_frente = 7
            data_alvo = hoje_data + timedelta(days=dias_a_frente)

        elif re.fullmatch(r"\d{1,2}", texto):
            dia = int(texto)
            try:
                candidata = hoje_data.replace(day=dia)
                # Se o dia já passou neste mês, projeta automaticamente para o próximo mês
                if candidata < hoje_data:
                    prox_mes = hoje_data.month + 1 if hoje_data.month < 12 else 1
                    prox_ano = hoje_data.year if hoje_data.month < 12 else hoje_data.year + 1
                    candidata = candidata.replace(year=prox_ano, month=prox_mes)
                data_alvo = candidata
            except ValueError:
                return False, data_texto, f"O dia '{data_texto}' não é válido no calendário."

        elif re.fullmatch(r"\d{1,2}/\d{1,2}", texto):
            dia, mes = map(int, texto.split("/"))
            try:
                candidata = hoje_data.replace(month=mes, day=dia)
                if candidata < hoje_data:
                    candidata = candidata.replace(year=hoje_data.year + 1)
                data_alvo = candidata
            except ValueError:
                return False, data_texto, f"A data '{data_texto}' não existe no calendário."

        elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", texto):
            try:
                data_alvo = datetime.strptime(texto, "%d/%m/%Y").date()
            except ValueError:
                return False, data_texto, f"A data '{data_texto}' é inválida."

    if data_alvo is None:
        return False, data_texto, f"Não consegui compreender a data '{data_texto}'."

    data_fmt = data_alvo.strftime("%d/%m/%Y")

    # Validação 1: Impede agendamento retroativo
    if data_alvo < hoje_data:
        return False, data_fmt, f"A data {data_fmt} já passou. Escolha uma data a partir de hoje ({hoje_data.strftime('%d/%m/%Y')})."

    # Validação 2: Oficina fechada aos domingos (weekday == 6)
    if data_alvo.weekday() == 6:
        return False, data_fmt, f"O dia {data_fmt} cai em um domingo e nossa oficina está fechada. Atendemos de segunda a sábado."

    return True, data_fmt, ""


def obter_horarios_ocupados(data_formatada: str) -> list[str]:
    """
    Lê agendamentos.txt e retorna os horários já confirmados para a data.
    """
    ocupados = []
    if not os.path.exists(ARQUIVO_AGENDAMENTOS):
        return ocupados

    with open(ARQUIVO_AGENDAMENTOS, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            if f"Data: {data_formatada}" in linha:
                busca_hora = re.search(r"Horário:\s*(\d{2}h)", linha)
                if busca_hora:
                    ocupados.append(busca_hora.group(1))

    return ocupados


def verificar_duplicidade_cliente(telefone: str, data_formatada: str) -> bool:
    """
    Verifica se o mesmo telefone já possui um agendamento confirmado nesta mesma data.
    """
    if not os.path.exists(ARQUIVO_AGENDAMENTOS):
        return False

    with open(ARQUIVO_AGENDAMENTOS, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            if f"Telefone: {telefone}" in linha and f"Data: {data_formatada}" in linha:
                return True
    return False


def ver_horarios_livres(data_formatada: str) -> list[str]:
    """
    Ferramenta simulada (RQ02).
    Retorna a grade de 1 em 1 hora (08h às 17h) descontando horários já agendados.
    """
    grade_base = ["08h", "09h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h"]
    ocupados = obter_horarios_ocupados(data_formatada)
    return [h for h in grade_base if h not in ocupados]


def sugerir_horario_proximo(horario_desejado: str, disponiveis: list[str]) -> str:
    """
    Sugere o próximo horário livre (intervalo de 1h) mais próximo do solicitado.
    """
    if not disponiveis:
        return "Nenhum horário livre nesta data"

    hora_alvo = int(horario_desejado.replace("h", ""))
    posteriores = [h for h in disponiveis if int(h.replace("h", "")) > hora_alvo]
    if posteriores:
        return posteriores[0]

    return min(disponiveis, key=lambda h: abs(int(h.replace("h", "")) - hora_alvo))


def exibir_grade_visual(data_label: str, horarios: list[str]) -> None:
    """
    Exibe uma tabela organizada no terminal com os horários livres da data.
    """
    tabela = Table(title=f"📅 Horários Livres - {data_label}", border_style="cyan")
    tabela.add_column("Manhã (08h - 11h)", justify="center", style="green")
    tabela.add_column("Tarde (12h - 17h)", justify="center", style="green")

    manha = [h for h in horarios if int(h.replace("h", "")) < 12]
    tarde = [h for h in horarios if int(h.replace("h", "")) >= 12]

    tabela.add_row(
        ", ".join(manha) if manha else "Lotado",
        ", ".join(tarde) if tarde else "Lotado",
    )
    console.print(tabela)


def salvar_agendamento_confirmado(
    protocolo: str,
    nome: str,
    telefone: str,
    veiculo: str,
    servico: str,
    data_agendada: str,
    horario: str,
) -> None:
    """
    Persiste o agendamento confirmado no arquivo principal agendamentos.txt.
    """
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linha = (
        f"[{timestamp}] Protocolo: {protocolo} | Cliente: {nome} | Telefone: {telefone} | "
        f"Veículo: {veiculo} | Data: {data_agendada} | Horário: {horario} | Serviço: {servico}\n"
    )
    with open(ARQUIVO_AGENDAMENTOS, "a", encoding="utf-8") as arquivo:
        arquivo.write(linha)


def realizar_atendimento() -> None:
    """
    Orquestra todo o fluxo do atendimento com validação de contato,
    auditoria em log e recuperação de agendamentos inacabados.
    """
    console.clear()
    protocolo = gerar_protocolo()
    saudacao = obter_saudacao()
    registrar_log(protocolo, "INICIO_SESSAO", f"Atendimento iniciado ({saudacao})")

    # Variáveis de estado do cliente para rastreamento e eventual salvamento em inacabados
    nome_cliente = "NAO_INFORMADO"
    telefone_cliente = "NAO_INFORMADO"
    telefone_valido = False
    veiculo_cliente = "NAO_INFORMADO"
    servico_desejado = "NAO_INFORMADO"
    data_salva = None
    ultimo_horario_tentado = "NAO_INFORMADO"

    console.print(
        Panel.fit(
            f"[bold cyan]🔧 OFICINA MECÂNICA AUTO FÁCIL 🔧[/bold cyan]\n"
            f"[white]{saudacao}! Seja muito bem-vindo ao nosso atendimento digital.[/white]\n"
            f"[dim]Protocolo de atendimento: {protocolo}[/dim]",
            border_style="blue",
        )
    )

    try:
        # =====================================================================
        # ETAPA 1: Nome do Cliente e Telefone de Contato (com validação Regex)
        # =====================================================================
        resposta_contato = Prompt.ask(
            f"\n[bold white]🤖 {saudacao}! Para iniciarmos seu atendimento, qual é o seu [bold cyan]nome[/bold cyan] e [bold cyan]telefone de contato[/bold cyan] (com ou sem DDD)?[/bold white]"
        )
        registrar_log(protocolo, "ENTRADA_CONTATO", f"Texto bruto: '{resposta_contato}'")

        with console.status("[bold green]Validando seus dados de contato...[/bold green]", spinner="dots"):
            try:
                extracao_contato = extrair_nome_telefone_com_ia(resposta_contato)
            except Exception as erro:
                console.print(f"[bold red]❌ Erro ao conectar com o Ollama:[/bold red] {erro}")
                registrar_log(protocolo, "ERRO_OLLAMA", str(erro))
                return

        nome_extraido = extracao_contato.get("nome", "NAO_INFORMADO")
        if nome_extraido != "NAO_INFORMADO" and nome_extraido.strip():
            nome_cliente = nome_extraido.strip()
        else:
            nome_cliente = Prompt.ask("[cyan]🤖 Como posso te chamar?[/cyan]").strip() or "Cliente"

        tel_bruto = extracao_contato.get("telefone", "NAO_INFORMADO")
        if tel_bruto != "NAO_INFORMADO":
            telefone_valido, telefone_formatado = validar_e_formatar_telefone(tel_bruto)
            if telefone_valido:
                telefone_cliente = telefone_formatado

        # Loop obrigatório até o telefone passar na validação Regex
        while not telefone_valido:
            registrar_log(protocolo, "TELEFONE_INVALIDO", f"Valor rejeitado: '{tel_bruto}'")
            tel_bruto = Prompt.ask(
                f"[yellow]⚠️ {nome_cliente}, não consegui validar um número de telefone válido.\n"
                "Por favor, digite seu telefone com ou sem DDD (Ex: (48) 99999-8888 ou 99999-8888):[/yellow]"
            )
            telefone_valido, telefone_formatado = validar_e_formatar_telefone(tel_bruto)
            if telefone_valido:
                telefone_cliente = telefone_formatado

        registrar_log(protocolo, "CONTATO_VALIDADO", f"Cliente: {nome_cliente} | Telefone: {telefone_cliente}")

        # =====================================================================
        # ETAPA 2: Modelo do Veículo e Serviço Desejado
        # =====================================================================
        resposta_veiculo_servico = Prompt.ask(
            f"\n[bold white]🤖 Perfeito, [bold cyan]{nome_cliente}[/bold cyan]! Qual é o [bold cyan]modelo do seu carro[/bold cyan] e qual [bold cyan]serviço ou reparo[/bold cyan] deseja executar?[/bold white]"
        )
        registrar_log(protocolo, "ENTRADA_VEICULO_SERVICO", f"Texto bruto: '{resposta_veiculo_servico}'")

        with console.status("[bold green]Analisando informações do veículo...[/bold green]", spinner="dots"):
            extracao_vs = extrair_veiculo_servico_com_ia(resposta_veiculo_servico)

        veiculo_extraido = extracao_vs.get("veiculo", "NAO_INFORMADO")
        servico_extraido = extracao_vs.get("servico", "NAO_INFORMADO")

        if veiculo_extraido != "NAO_INFORMADO" and veiculo_extraido.strip():
            veiculo_cliente = veiculo_extraido.strip()
        else:
            veiculo_cliente = Prompt.ask("[cyan]🤖 Qual é o modelo/ano do veículo?[/cyan]").strip() or "Não informado"

        if servico_extraido != "NAO_INFORMADO" and servico_extraido.strip():
            servico_desejado = servico_extraido.strip()
        else:
            servico_desejado = Prompt.ask(f"[cyan]🤖 Qual serviço ou defeito o {veiculo_cliente} apresenta?[/cyan]").strip() or "Revisão Geral"

        registrar_log(protocolo, "VEICULO_SERVICO_REGISTRADO", f"Veículo: {veiculo_cliente} | Serviço: {servico_desejado}")

        # =====================================================================
        # ETAPA 3: Dia e Horário (Validação, Sugestão e Limite de 3 Tentativas)
        # =====================================================================
        max_tentativas = 3
        tentativa_atual = 0

        console.print(
            f"\n[green]🤖 Anotado! Vamos agendar o serviço de [bold]{servico_desejado}[/bold] "
            f"para o seu [bold]{veiculo_cliente}[/bold].[/green]"
        )

        while tentativa_atual < max_tentativas:
            tentativa_atual += 1
            console.print(f"\n[bold yellow]► Escolha de Agenda (Tentativa {tentativa_atual} de {max_tentativas})[/bold yellow]")

            if data_salva:
                pergunta_agenda = (
                    f"[bold white]🤖 Mantendo a data [bold cyan]{data_salva}[/bold cyan], "
                    f"qual outro horário você prefere?[/bold white]"
                )
            else:
                pergunta_agenda = (
                    "[bold white]🤖 Para qual [bold cyan]dia e horário[/bold cyan] gostaria de agendar? "
                    "[dim](Ex: Hoje às 14h, Amanhã às 10h, Sábado às 09h ou dia 28 às 15h)[/dim][/bold white]"
                )

            mensagem_agenda = Prompt.ask(pergunta_agenda)
            registrar_log(protocolo, f"TENTATIVA_{tentativa_atual}", f"Entrada agenda: '{mensagem_agenda}'")

            if not mensagem_agenda.strip():
                console.print("[yellow]⚠️ Mensagem vazia! Informe o dia e o horário desejado.[/yellow]")
                continue

            with console.status("[bold green]🤖 Consultando disponibilidade na oficina...[/bold green]", spinner="dots"):
                extracao_dh = extrair_data_horario_com_ia(mensagem_agenda)

            registrar_log(protocolo, f"IA_EXTRACAO_{tentativa_atual}", f"JSON extraído: {json.dumps(extracao_dh, ensure_ascii=False)}")

            # Resolve e valida a data (se informou nova data, valida; se não, mantém a anterior ou Hoje)
            texto_data_bruta = extracao_dh["data"] if extracao_dh["data"] != "NAO_INFORMADA" else (data_salva or "Hoje")
            data_ok, data_calculada, msg_erro_data = resolver_data_segura(texto_data_bruta)

            if not data_ok:
                console.print(f"[bold red]❌ {msg_erro_data}[/bold red]")
                registrar_log(protocolo, "DATA_INVALIDA", msg_erro_data)
                continue

            data_salva = data_calculada
            horario_pedido = extracao_dh["horario"]

            if horario_pedido == "NENHUM":
                console.print(
                    f"[yellow]🤔 Entendi a data [bold]{data_salva}[/bold], mas faltou informar o horário exato (ex: 09h, 14h, 16h).[/yellow]"
                )
                continue

            ultimo_horario_tentado = horario_pedido

            # Verifica se o mesmo telefone já possui agendamento nesta data
            if verificar_duplicidade_cliente(telefone_cliente, data_salva):
                console.print(
                    Panel(
                        f"[bold yellow]⚠️ Atenção, {nome_cliente}![/bold yellow]\n"
                        f"Já consta um agendamento confirmado para o telefone [bold]{telefone_cliente}[/bold] "
                        f"no dia [bold]{data_salva}[/bold].\n"
                        "Caso deseje marcar outro serviço, escolha uma nova data.",
                        title="Agendamento Duplicado Detectado",
                        border_style="yellow",
                    )
                )
                registrar_log(protocolo, "DUPLICIDADE_TELEFONE", f"Tel {telefone_cliente} já agendado em {data_salva}")
                data_salva = None
                continue

            # RQ02 e RQ03: Consulta a ferramenta e valida disponibilidade
            horarios_disponiveis = ver_horarios_livres(data_salva)
            horarios_ocupados = obter_horarios_ocupados(data_salva)

            if horario_pedido not in horarios_disponiveis:
                sugestao = sugerir_horario_proximo(horario_pedido, horarios_disponiveis)
                motivo_indisp = (
                    f"O horário das [bold]{horario_pedido}[/bold] já está reservado"
                    if horario_pedido in horarios_ocupados
                    else f"O horário das [bold]{horario_pedido}[/bold] está fora do expediente (08h às 17h)"
                )

                console.print(
                    Panel(
                        f"[bold red]❌ {motivo_indisp} no dia {data_salva}.[/bold red]\n"
                        f"📌 Deixei o dia [bold cyan]{data_salva}[/bold cyan] reservado para você.\n"
                        f"💡 [bold green]Sugestão mais próxima (intervalo de 1h):[/bold green] "
                        f"Temos vaga às [bold underline]{sugestao}[/bold underline]!",
                        title="Horário Indisponível",
                        border_style="red",
                    )
                )
                exibir_grade_visual(data_salva, horarios_disponiveis)
                registrar_log(protocolo, "HORARIO_INDISPONIVEL", f"{data_salva} às {horario_pedido} | Sugestão: {sugestao}")
                continue

            # Critério 6: Resumo completo para validação humana da equipe
            resumo_texto = (
                f"[bold]• Protocolo:[/bold] {protocolo}\n"
                f"[bold]• Cliente:[/bold] {nome_cliente}\n"
                f"[bold]• Telefone:[/bold] {telefone_cliente}\n"
                f"[bold]• Veículo:[/bold] {veiculo_cliente}\n"
                f"[bold]• Serviço / Defeito:[/bold] {servico_desejado}\n"
                f"[bold]• Data:[/bold] {data_salva}\n"
                f"[bold]• Horário:[/bold] {horario_pedido} (Duração: 1 hora)"
            )
            console.print("\n", Panel(resumo_texto, title="📋 RESUMO DO AGENDAMENTO PARA A EQUIPE", border_style="green"))

            # RQ04: Confirmação humana obrigatória antes de salvar
            confirmacao = Prompt.ask(
                "[bold yellow]Confirmar agendamento? (S/N)[/bold yellow]",
                choices=["S", "N", "s", "n"],
            ).upper()

            if confirmacao == "S":
                salvar_agendamento_confirmado(
                    protocolo,
                    nome_cliente,
                    telefone_cliente,
                    veiculo_cliente,
                    servico_desejado,
                    data_salva,
                    horario_pedido,
                )
                registrar_log(protocolo, "AGENDAMENTO_CONFIRMADO", f"{data_salva} às {horario_pedido} salvo em {ARQUIVO_AGENDAMENTOS}")

                console.print(
                    Panel(
                        f"[bold green]✅ Agendamento confirmado com sucesso para o {veiculo_cliente} "
                        f"no dia {data_salva} às {horario_pedido}![/bold green]\n"
                        f"[dim](Protocolo {protocolo} salvo em '{ARQUIVO_AGENDAMENTOS}')[/dim]\n\n"
                        f"[white]Muito obrigado pela preferência, [bold cyan]{nome_cliente}[/bold cyan]! "
                        f"Nossa equipe estará aguardando por você.\n"
                        f"📞 Caso ocorra algum imprevisto e não consiga comparecer, por favor, "
                        f"entre em contato conosco com antecedência para reagendarmos. Até breve![/white]",
                        title="🤝 Atendimento Concluído",
                        border_style="green",
                    )
                )
                return
            else:
                console.print("\n[bold red]🚫 Agendamento cancelado pela equipe. Os dados foram salvos para retorno posterior.[/bold red]")
                registrar_log(protocolo, "CANCELADO_PELA_EQUIPE", "Operador respondeu N")
                salvar_agendamento_inacabado(
                    protocolo,
                    nome_cliente,
                    telefone_cliente,
                    veiculo_cliente,
                    servico_desejado,
                    data_salva or "NAO_INFORMADA",
                    ultimo_horario_tentado,
                    "Cancelado pela equipe (N)",
                )
                return

        # RQ05: Limite de 3 tentativas atingido -> salva em inacabados para o atendente ligar
        salvar_agendamento_inacabado(
            protocolo,
            nome_cliente,
            telefone_cliente,
            veiculo_cliente,
            servico_desejado,
            data_salva or "NAO_INFORMADA",
            ultimo_horario_tentado,
            "Limite de 3 tentativas atingido",
        )
        console.print(
            Panel(
                f"[bold yellow]🙏 Agradecemos o contato, {nome_cliente}![/bold yellow]\n"
                "Como atingimos o limite de 3 tentativas de escolha de horário, encerramos esta etapa.\n"
                f"📞 Não se preocupe: registramos seu telefone ([bold]{telefone_cliente}[/bold]) e "
                "um de nossos atendentes entrará em contato em breve para concluir seu agendamento!",
                border_style="yellow",
            )
        )

    except KeyboardInterrupt:
        # Captura Ctrl+C: se o cliente já informou nome e telefone, salva em inacabados antes de sair
        console.print("\n[bold yellow]⚠️ Atendimento interrompido pelo usuário.[/bold yellow]")
        if nome_cliente != "NAO_INFORMADO" and telefone_valido:
            salvar_agendamento_inacabado(
                protocolo,
                nome_cliente,
                telefone_cliente,
                veiculo_cliente,
                servico_desejado,
                data_salva or "NAO_INFORMADA",
                ultimo_horario_tentado,
                "Interrupção abrupta (Ctrl+C)",
            )
            console.print(f"[dim]Dados parciais salvos em '{ARQUIVO_INACABADOS}' para retorno da equipe.[/dim]")
        registrar_log(protocolo, "INTERRUPCAO_USUARIO", "Execução abortada via KeyboardInterrupt")


def executar_agente() -> None:
    """
    Loop externo que permite realizar múltiplos atendimentos em sequência.
    """
    while True:
        realizar_atendimento()
        try:
            novo = Prompt.ask(
                "\n[bold cyan]🔄 Deseja fazer um novo agendamento? (S/N)[/bold cyan]",
                choices=["S", "N", "s", "n"],
            ).upper()
        except KeyboardInterrupt:
            break

        if novo != "S":
            console.print(
                Panel(
                    "[bold green]👋 Obrigado por utilizar o assistente da Oficina Mecânica Auto Fácil![/bold green]\n"
                    "[white]Lembre-se: caso não consiga comparecer no horário agendado, entre em contato conosco.\n"
                    "Tenha um excelente dia e até a próxima![/white]",
                    border_style="blue",
                )
            )
            break


if __name__ == "__main__":
    executar_agente()