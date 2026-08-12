import os
import gradio as gr

# ==========================================
# 1. SUA LÓGICA E FUNÇÕES PYTHON
# ==========================================
def minha_funcao(texto):
    if not texto:
        return "Por favor, insira algum texto para processar."
    return f"Resultado processado: {texto.upper()}"


# ==========================================
# 2. INTERFACE GRADIO
# ==========================================
with gr.Blocks(title="Painel Gerencial") as demo:
    gr.Markdown("# 📊 Painel Gerencial")
    gr.Markdown("Seja bem-vindo ao sistema. Preencha os campos abaixo para testar.")

    with gr.Row():
        entrada = gr.Textbox(label="Dados de Entrada", placeholder="Digite aqui...")
        saida = gr.Textbox(label="Resultado", interactive=False)

    botao = gr.Button("Processar", variant="primary")
    botao.click(fn=minha_funcao, inputs=entrada, outputs=saida)


# ==========================================
# 3. CONFIGURAÇÃO DE SERVIDOR (OBRIGATÓRIO PARA O RENDER)
# ==========================================
if __name__ == "__main__":
    # O Render atribui uma porta dinâmica na variável PORT
    port = int(os.environ.get("PORT", 7860))

    # server_name "0.0.0.0" permite conexões externas do Render
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False  # Não use share=True no Render
    )
