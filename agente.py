import re
from datetime import datetime
import ollama

# Configuração do modelo gratuito e local exigido no projeto
MODELO_LLM = "qwen2.5:1.5b"
ARQUIVO_SAIDA = "agendamentos.txt"


def ver_horarios_livres() -> list[str]:
    """
    Ferramenta simulada (RQ02).
    Retorna a lista fixa de horários disponíveis na oficina Auto Fácil.
    """
    return ["08h", "10h", "14h", "16h"]


def extrair_horario_com_ia(mensagem_cliente: str) -> str:
    """
    Envia a mensagem do cliente para o LLM local (RQ01) e extrai
    apenas o horário solicitado no formato 'XXh' (ex: '14h').
    """
    prompt_sistema = (
        "Você é um extrator de horários da oficina Auto Fácil. "
        "Sua única tarefa é identificar o horário pedido pelo cliente e responder "
        "EXCLUSIVAMENTE no formato XXh (dois dígitos seguidos da letra h). "
        "Não escreva frases, saudações ou explicações. "
        "Se não houver horário na mensagem, responda exatamente: NENHUM.\n\n"
        "Exemplos:\n"
        "Cliente: Quero marcar uma revisão às 14h -> 14h\n"
        "Cliente: Tem vaga amanhã as 8 da manhã? -> 08h\n"
        "Cliente: Pode ser 16:00? -> 16h\n"
        "Cliente: Gostaria de olhar os freios -> NENHUM"
    )

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": mensagem_cliente},
        ],
        # Temperatura 0 deixa o modelo determinístico e evita criatividade indesejada
        options={"temperature": 0.0},
    )

    texto_bruto = resposta["message"]["content"].strip()

    # Limpeza preventiva caso o modelo pequeno adicione algum texto extra junto ao horário
    busca = re.search(r"\b(\d{1,2})h\b", texto_bruto, re.IGNORECASE)
    if busca:
        hora_formatada = busca.group(1).zfill(2) + "h"
        return hora_formatada.lower()

    return "NENHUM"


def salvar_agendamento(horario: str, mensagem_original: str) -> None:
    """
    Persiste o agendamento confirmado no arquivo agendamentos.txt.
    """
    data_registro = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linha = f"[{data_registro}] Horário: {horario} | Pedido: \"{mensagem_original}\"\n"

    with open(ARQUIVO_SAIDA, "a", encoding="utf-8") as arquivo:
        arquivo.write(linha)


def executar_agente() -> None:
    """
    Orquestrador principal do agente com controle de tentativas,
    validação de ferramenta e confirmação humana (Human-in-the-Loop).
    """
    print("=" * 55)
    print("🔧 ASSISTENTE DE AGENDAMENTO - OFICINA AUTO FÁCIL 🔧")
    print("=" * 55)

    max_tentativas = 3
    tentativa_atual = 0

    # RQ05: Controle de execução com limite máximo de 3 tentativas
    while tentativa_atual < max_tentativas:
        tentativa_atual += 1
        print(f"\n[Tentativa {tentativa_atual} de {max_tentativas}]")
        mensagem = input("Cliente, como posso ajudar com sua revisão? ")

        if not mensagem.strip():
            print("⚠️ Ops! Você enviou uma mensagem vazia. Por favor, informe o horário desejado.")
            continue

        print("🤖 Analisando pedido com a IA...")
        try:
            horario_pedido = extrair_horario_com_ia(mensagem)
        except Exception as erro:
            print(f"❌ Erro ao conectar com o Ollama: {erro}")
            print("Verifique se o Ollama está rodando e se o modelo qwen2.5:1.5b foi baixado.")
            return

        # Verifica se a IA conseguiu identificar algum horário na frase
        if horario_pedido == "NENHUM":
            print("🤔 Não consegui identificar um horário específico na sua mensagem.")
            print("💡 Dica: Tente algo como 'Quero marcar uma revisão às 14h'.")
            continue

        print(f"🔎 Horário identificado pela IA: {horario_pedido}")

        # RQ02 e RQ03: Consulta a ferramenta e valida se o horário existe na lista
        horarios_disponiveis = ver_horarios_livres()

        if horario_pedido not in horarios_disponiveis:
            print(f"❌ Desculpe, o horário das {horario_pedido} está ocupado ou não existe na nossa grade.")
            print(f"📅 Horários livres disponíveis hoje: {', '.join(horarios_disponiveis)}")
            continue

        # Critério 6: Exibir resumo claro antes de pedir confirmação humana (S/N)
        print("\n" + "-" * 45)
        print("📋 RESUMO DO AGENDAMENTO PARA A EQUIPE")
        print(f"   • Serviço: Revisão Veicular")
        print(f"   • Horário Solicitado: {horario_pedido}")
        print(f"   • Mensagem Original: \"{mensagem}\"")
        print("-" * 45)

        # RQ04: Confirmação humana obrigatória antes de salvar
        confirmacao = input("Confirmar agendamento? (S/N): ").strip().upper()

        if confirmacao == "S":
            salvar_agendamento(horario_pedido, mensagem)
            print(f"✅ Sucesso! Agendamento das {horario_pedido} confirmado e salvo em '{ARQUIVO_SAIDA}'.")
            return
        else:
            print("🚫 Agendamento cancelado pela equipe. Nenhum dado foi salvo.")
            return

    # Mensagem amigável ao atingir o limite de 3 tentativas (RQ05)
    print("\n" + "=" * 55)
    print("🙏 Agradecemos o contato com a Oficina Auto Fácil!")
    print("Como atingimos o limite de 3 tentativas, encerramos este atendimento.")
    print("Se preferir, ligue diretamente para nossa recepção. Até logo!")
    print("=" * 55)


if __name__ == "__main__":
    executar_agente()