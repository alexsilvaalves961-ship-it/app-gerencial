import os
import gradio as gr


def processar_dados(texto):
    if not texto:
        return "Por favor, digite algo para testar."
    return f"Processado com sucesso: {texto.upper()}"


with gr.Blocks(title="Sistema Gerencial") as demo:
    gr.Markdown("# 📊 Painel Gerencial")
    gr.Markdown("Aplicação online pronta para testes.")

    with gr.Row():
        entrada = gr.Textbox(
            label="Entrada de Dados", placeholder="Digite aqui..."
        )
        saida = gr.Textbox(label="Resultado", interactive=False)

    botao = gr.Button("Executar", variant="primary")
    botao.click(fn=processar_dados, inputs=entrada, outputs=saida)


port = int(os.environ.get("PORT", 10000))

demo.launch(server_name="0.0.0.0", server_port=port, share=False)
