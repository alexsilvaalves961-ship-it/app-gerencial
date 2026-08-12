import os
import gradio as gr

# --- SEU CÓDIGO E FUNÇÕES AQUI ---

with gr.Blocks() as demo:
    gr.Markdown("# Painel de Controle")
    # Suas abas, botões e entradas do Gradio entram aqui...

# --- CONFIGURAÇÃO OBRIGATÓRIA PARA O RENDER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)