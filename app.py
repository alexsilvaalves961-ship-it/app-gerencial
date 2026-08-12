import os
import gradio as gr

# ==========================================
# 1. SUAS FUNÇÕES / LÓGICA DO PROGRAMA
# ==========================================
def processar_dados(texto):
    if not texto:
        return "Por favor, digite algo para testar."
    return f"Processado com sucesso: {texto.upper()}"


# ==========================================
# 2. CONSTRUÇÃO DA INTERFACE (GRADIO)
# ==========================================
with gr.Blocks(title="Sistema Gerencial") as demo:
    gr.Markdown("# 📊 Painel Gerencial")
    gr.Markdown("Aplicação online pronta para testes.")

    with gr.Row():
        entrada = gr.Textbox(label="Entrada de Dados", placeholder="Digite aqui...")
        saida = gr.Textbox(label="Resultado", interactive=False)

    botao = gr.Button("Executar", variant="primary")
    botao.click(fn=processar_dados, inputs=entrada, outputs=saida)


# ==========================================
# 3. INICIALIZAÇÃO OBRIGATÓRIA PARA O RENDER
# ==========================================
# Captura a porta dinâmica atribuída pelo Render (padrão 10000)
port = int(os.environ.get("PORT", 10000))

# Inicializa o servidor diretamente para garantir que o Render abra a porta
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False
)
