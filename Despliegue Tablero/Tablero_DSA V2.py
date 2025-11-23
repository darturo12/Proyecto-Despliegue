from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests
import json

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# URL de tu API en Railway (REEMPLAZA CON TU URL REAL)
API_URL = "https://prueba-api-production-21a9.up.railway.app/api/v1/predict"

COLORS = {
    'primary': '#1e3a8a',
    'secondary': '#3b82f6',
    'accent': '#10b981',
    'background': '#f8fafc',
    'card': '#ffffff',
    'text': '#1e293b',
    'text_light': '#64748b',
    'border': '#e2e8f0'
}

card_style = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '12px',
    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
    'padding': '24px',
    'border': f'1px solid {COLORS["border"]}'
}

header_style = {
    'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
    'color': 'white',
    'padding': '32px',
    'borderRadius': '12px',
    'marginBottom': '32px',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
}


def crear_grupo_botones(titulo, opciones, grupo_id):
    return dbc.Col([
        html.Label(titulo, className="fw-semibold mb-2", style={'color': COLORS['text'], 'fontSize': '13px'}),
        dbc.ButtonGroup(
            [dbc.Button(
                op, 
                id=f"{grupo_id}_{op.replace(' ', '_').replace('-', '_')}", 
                outline=True, 
                color="primary",
                size="sm",
                className="btn-toggle"
            ) for op in opciones],
            className="d-flex flex-wrap gap-1"
        ),
    ], lg=3, md=4, sm=6, xs=12, className="mb-3")


def crear_slider_section(label, id_base, min_val, max_val, step, default, marks):
    return dbc.Col([
        html.Div([
            html.Label(label, className="fw-semibold mb-1", style={'color': COLORS['text'], 'fontSize': '13px'}),
            html.Span(id=f"{id_base}-val", className="badge bg-primary ms-2", style={'fontSize': '14px'}),
        ], className="d-flex align-items-center mb-2"),
        dcc.Slider(
            min_val, max_val, step, 
            value=default, 
            marks=marks, 
            id=f"{id_base}-slider",
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], lg=4, md=6, xs=12, className="mb-4")


def crear_grafico(x_data, y_data, titulo, x_label, y_label):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        line=dict(color=COLORS['accent'], width=3, shape='spline'),
        marker=dict(size=8, color=COLORS['accent'], line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor=f'rgba(16, 185, 129, 0.1)',
        hovertemplate='<b>%{x}</b><br>Probabilidad: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16, color=COLORS['text'], family='Arial, sans-serif')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=320,
        margin=dict(l=50, r=30, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_light'], size=12),
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor=COLORS['border'], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=COLORS['border'], zeroline=False, range=[0, 1])
    )
    
    return fig


app.layout = dbc.Container([
    # Stores para guardar valores seleccionados
    dcc.Store(id='empleo-seleccionado', data='management'),
    dcc.Store(id='civil-seleccionado', data='married'),
    dcc.Store(id='educacion-seleccionado', data='tertiary'),
    dcc.Store(id='creditos-seleccionado', data='no'),
    dcc.Store(id='hipotecario-seleccionado', data='no'),
    dcc.Store(id='personal-seleccionado', data='no'),
    dcc.Store(id='contacto-seleccionado', data='cellular'),
    dcc.Store(id='resultado-seleccionado', data='unknown'),
    dcc.Store(id='mes-seleccionado', data='may'),
    dcc.Store(id='ultimo-dia-contacto', data=15),
   
    # Header
    html.Div([
        html.H1("Tablero de Preevaluación de Campañas", 
                className="mb-2 fw-bold", 
                style={'fontSize': '32px', 'letterSpacing': '-0.5px'}),
        html.P("Sistema de análisis predictivo para optimización de campañas de marketing", 
               className="mb-0 opacity-90", 
               style={'fontSize': '15px'})
    ], style=header_style),

    # Características del Cliente
    html.Div([
        html.H5("Características del Cliente Tipo", 
                className="mb-4 fw-semibold d-flex align-items-center",
                style={'color': COLORS['primary']}),
        
        html.H6("Información Demográfica", className="text-muted mb-3 mt-2", style={'fontSize': '14px'}),
        dbc.Row([
            crear_grupo_botones("Tipo de Empleo", [
                "management", "technician", "entrepreneur", "blue-collar", "retired", "services",
                "admin", "self-employed", "unemployed", "housemaid", "student", "unknown"
            ], "empleo"),
            crear_grupo_botones("Estado Civil", ["married", "single", "divorced"], "civil"),
            crear_grupo_botones("Nivel Educativo", ["primary", "secondary", "tertiary", "unknown"], "educacion"),
        ]),
        
        html.Hr(className="my-4", style={'borderColor': COLORS['border']}),
        
        html.H6("Información Financiera y de Contacto", className="text-muted mb-3", style={'fontSize': '14px'}),
        dbc.Row([
            crear_grupo_botones("Productos Crediticios", ["yes", "no"], "creditos"),
            crear_grupo_botones("Crédito Hipotecario", ["yes", "no"], "hipotecario"),
            crear_grupo_botones("Crédito Personal", ["yes", "no"], "personal"),
            crear_grupo_botones("Modo de Contacto", ["cellular", "telephone", "unknown"], "contacto"),
        ]),
        
        html.Hr(className="my-4", style={'borderColor': COLORS['border']}),
        
        html.H6("Información de Campaña", className="text-muted mb-3", style={'fontSize': '14px'}),
        dbc.Row([
            crear_grupo_botones("Resultado Última Campaña", ["success", "failure", "other", "unknown"], "resultado"),
            crear_grupo_botones("Mes de Contacto", [
                "jan", "feb", "mar", "apr", "may", "jun", 
                "jul", "aug", "sep", "oct", "nov", "dec"
            ], "mes"),
        ]),
    ], style=card_style, className="mb-4"),

    # Variables Continuas
    html.Div([
        html.H5("Variables Continuas", 
                className="mb-4 fw-semibold",
                style={'color': COLORS['primary']}),
        
        dbc.Row([
            crear_slider_section("Edad", "edad", 18, 95, 1, 45, {18: "18", 40: "40", 60: "60", 80: "80", 95: "95"}),
            crear_slider_section("Balance ($)", "bal", -8000, 110000, 1000, 570, 
                               {-8000: "-8K", 0: "0", 50000: "50K", 110000: "110K"}),
            crear_slider_section("Duración (seg)", "dur", 0, 3500, 50, 1800, 
                               {0: "0", 1000: "1K", 2000: "2K", 3500: "3.5K"}),
        ]),
        
        dbc.Row([
            crear_slider_section("Campaña", "camp", 1, 60, 1, 2, {1: "1", 20: "20", 40: "40", 60: "60"}),
            crear_slider_section("Días desde Último Contacto", "dias", -1, 871, 10, -1, 
                               {-1: "Nunca", 0: "0", 200: "200", 400: "400", 871: "871"}),
            crear_slider_section("Contactos Previos", "cont", 0, 275, 5, 0, 
                               {0: "0", 90: "90", 180: "180", 275: "275"}),
        ]),
        
        dbc.Row([
            crear_slider_section("Día del Mes del Último Contacto", "dia", 1, 31, 1, 15,
                               {1: "1", 10: "10", 20: "20", 31: "31"}),
        ]),
    ], style=card_style, className="mb-4"),

    # Botón Evaluar
    html.Div([
        dbc.Button(
            [
                html.I(className="bi bi-calculator me-2"),
                "Evaluar"
            ],
            id="btn-evaluar",
            size="lg",
            className="px-5 py-3 fw-semibold",
            style={
                'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                'border': 'none',
                'borderRadius': '12px',
                'fontSize': '18px',
                'boxShadow': '0 4px 12px rgba(30, 58, 138, 0.3)',
                'transition': 'all 0.3s ease'
            }
        )
    ], className="text-center mb-4"),

    # Resultados
    html.Div([
        html.H5("Análisis de Resultados", 
                className="mb-4 fw-semibold",
                style={'color': COLORS['primary']}),
        
        dbc.Row([
            # Card de probabilidad
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(id="icono-resultado", className="bi bi-graph-up-arrow", 
                               style={'fontSize': '40px', 'color': COLORS['accent']}),
                    ], className="text-center mb-3"),
                    html.H1(id="probabilidad-texto", children="--", 
                            className="text-center fw-bold mb-2", 
                            style={'color': COLORS['primary'], 'fontSize': '48px'}),
                    html.P(id="probabilidad-descripcion", children="Presione 'Evaluar' para calcular", 
                           className="text-center mb-0 text-muted", 
                           style={'fontSize': '14px', 'fontWeight': '500'}),
                    html.Div([
                        html.Span(id="probabilidad-badge", children="", 
                                 className="badge mt-2", style={'fontSize': '12px'})
                    ], className="text-center")
                ], style={
                    'backgroundColor': f'{COLORS["background"]}',
                    'borderRadius': '12px',
                    'padding': '24px',
                    'height': '100%',
                    'border': f'2px solid {COLORS["border"]}'
                })
            ], lg=3, md=12, className="mb-3"),
            
            # Gráficos
            dbc.Col([
                dcc.Graph(
                    id="grafico-edad",
                    figure=crear_grafico(
                        [18, 30, 45, 60, 80],
                        [0.2, 0.3, 0.5, 0.7, 0.8],
                        "Probabilidad vs Edad",
                        "Edad (años)",
                        "P(aceptación)"
                    ),
                    config={"displayModeBar": False}
                )
            ], lg=4, md=6, className="mb-3"),
            
            dbc.Col([
                dcc.Graph(
                    id="grafico-balance",
                    figure=crear_grafico(
                        [-8000, 0, 20000, 60000, 110000],
                        [0.8, 0.6, 0.4, 0.6, 0.8],
                        "Probabilidad vs Balance",
                        "Balance ($)",
                        "P(aceptación)"
                    ),
                    config={"displayModeBar": False}
                )
            ], lg=5, md=6, className="mb-3"),
        ]),
    ], style=card_style, className="mb-4"),

    # Recomendaciones
    html.Div(id="recomendaciones-container", children=[
        html.Div([
            html.I(className="bi bi-lightbulb-fill me-2", style={'color': COLORS['accent']}),
            html.Span("Recomendaciones Estratégicas", className="fw-semibold")
        ], className="d-flex align-items-center mb-3", style={'color': COLORS['primary'], 'fontSize': '16px'}),
        
        html.P("Presione 'Evaluar' para obtener recomendaciones personalizadas basadas en el perfil del cliente.",
               className="mb-0", style={'color': COLORS['text']})
    ], style={**card_style, 'backgroundColor': f'{COLORS["background"]}', 'borderLeft': f'4px solid {COLORS["accent"]}'}, 
       className="mb-4"),

], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '32px', 'minHeight': '100vh'})


# ==================== CALLBACKS DE SLIDERS ====================
@app.callback(Output("edad-val", "children"), Input("edad-slider", "value"))
def update_edad(v): return f"{v} años"

@app.callback(Output("bal-val", "children"), Input("bal-slider", "value"))
def update_bal(v): return f"${v:,}"

@app.callback(Output("dur-val", "children"), Input("dur-slider", "value"))
def update_dur(v): return f"{v} seg"

@app.callback(Output("camp-val", "children"), Input("camp-slider", "value"))
def update_camp(v): return v

@app.callback(Output("dias-val", "children"), Input("dias-slider", "value"))
def update_dias(v): return f"{v} días" if v >= 0 else "Nunca contactado"

@app.callback(Output("cont-val", "children"), Input("cont-slider", "value"))
def update_cont(v): return v

@app.callback(Output("dia-val", "children"), Input("dia-slider", "value"))
def update_dia(v): return f"Día {v}"


# ==================== CALLBACKS DE BOTONES ====================
def crear_callback_botones_store(grupo_id, opciones, store_id):
    """Crea callback para un grupo de botones que actualiza un store"""
    outputs = [Output(f"{grupo_id}_{op.replace(' ', '_').replace('-', '_')}", "outline") for op in opciones]
    outputs += [Output(f"{grupo_id}_{op.replace(' ', '_').replace('-', '_')}", "color") for op in opciones]
    outputs.append(Output(store_id, "data"))
    
    inputs = [Input(f"{grupo_id}_{op.replace(' ', '_').replace('-', '_')}", "n_clicks") for op in opciones]
    
    @app.callback(outputs, inputs)
    def toggle_buttons(*args):
        if not ctx.triggered_id:
            # Valores iniciales: primer botón seleccionado
            return [False] + [True] * (len(opciones) - 1) + ["primary"] * len(opciones) + [opciones[0]]
        
        clicked_id = ctx.triggered_id
        results_outline = []
        results_color = []
        selected_value = None
        
        for op in opciones:
            button_id = f"{grupo_id}_{op.replace(' ', '_').replace('-', '_')}"
            if button_id == clicked_id:
                results_outline.append(False)  
                results_color.append("primary")
                selected_value = op
            else:
                results_outline.append(True)   
                results_color.append("primary")
        
        return results_outline + results_color + [selected_value]
    
    return toggle_buttons


# Crear callbacks para cada grupo
crear_callback_botones_store("empleo", [
    "management", "technician", "entrepreneur", "blue-collar", "retired", "services",
    "admin", "self-employed", "unemployed", "housemaid", "student", "unknown"
], "empleo-seleccionado")

crear_callback_botones_store("civil", ["married", "single", "divorced"], "civil-seleccionado")

crear_callback_botones_store("educacion", ["primary", "secondary", "tertiary", "unknown"], "educacion-seleccionado")

crear_callback_botones_store("creditos", ["yes", "no"], "creditos-seleccionado")

crear_callback_botones_store("hipotecario", ["yes", "no"], "hipotecario-seleccionado")

crear_callback_botones_store("personal", ["yes", "no"], "personal-seleccionado")

crear_callback_botones_store("contacto", ["cellular", "telephone", "unknown"], "contacto-seleccionado")

crear_callback_botones_store("resultado", ["success", "failure", "other", "unknown"], "resultado-seleccionado")

crear_callback_botones_store("mes", [
    "jan", "feb", "mar", "apr", "may", "jun", 
    "jul", "aug", "sep", "oct", "nov", "dec"
], "mes-seleccionado")


# ==================== CALLBACK PRINCIPAL DE PREDICCIÓN ====================
@app.callback(
    [
        Output("probabilidad-texto", "children"),
        Output("probabilidad-descripcion", "children"),
        Output("probabilidad-badge", "children"),
        Output("icono-resultado", "className"),
        Output("icono-resultado", "style"),
        Output("recomendaciones-container", "children"),
    ],
    Input("btn-evaluar", "n_clicks"),
    [
        # Sliders
        State("edad-slider", "value"),
        State("bal-slider", "value"),
        State("dur-slider", "value"),
        State("camp-slider", "value"),
        State("dias-slider", "value"),
        State("cont-slider", "value"),
        State("dia-slider", "value"),
        # Stores
        State("empleo-seleccionado", "data"),
        State("civil-seleccionado", "data"),
        State("educacion-seleccionado", "data"),
        State("creditos-seleccionado", "data"),
        State("hipotecario-seleccionado", "data"),
        State("personal-seleccionado", "data"),
        State("contacto-seleccionado", "data"),
        State("resultado-seleccionado", "data"),
        State("mes-seleccionado", "data"),
    ],
    prevent_initial_call=True
)
def realizar_prediccion(n_clicks, edad, balance, duracion, campana, dias, contactos, dia,
                       empleo, civil, educacion, creditos, hipotecario, personal,
                       contacto, resultado, mes):
    
    if not n_clicks:
        return "--", "Presione 'Evaluar' para calcular", "", "bi bi-graph-up-arrow", {'fontSize': '40px', 'color': COLORS['accent']}, []
    
    # Preparar datos para la API
    datos = {
        "inputs": [{
            "Age": edad,
            "Job": empleo,
            "Marital.Status": civil,
            "Education": educacion,
            "Credit": creditos,
            "Balance..euros.": balance,
            "Housing.Loan": hipotecario,
            "Personal.Loan": personal,
            "Contact": contacto,
            "Last.Contact.Day": dia,
            "Last.Contact.Month": mes,
            "Last.Contact.Duration": duracion,
            "Campaign": campana,
            "Pdays": dias,
            "Previous": contactos,
            "Poutcome": resultado
        }]
    }
    
    try:
        # Hacer petición a la API
        response = requests.post(API_URL, json=datos, timeout=30)
        response.raise_for_status()
        
        # Procesar respuesta
        resultado_api = response.json()
        prediccion = resultado_api['predictions'][0]
        
        # Interpretar resultado
        if prediccion == 1:
            prob_texto = "Alta"
            prob_desc = "Probabilidad de Suscripción"
            badge_text = "✓ Suscripción Probable"
            badge_class = "badge bg-success mt-2"
            icono_class = "bi bi-check-circle-fill"
            icono_style = {'fontSize': '40px', 'color': '#10b981'}
            
            recomendaciones = [
                html.Div([
                    html.I(className="bi bi-lightbulb-fill me-2", style={'color': COLORS['accent']}),
                    html.Span("Recomendaciones Estratégicas", className="fw-semibold")
                ], className="d-flex align-items-center mb-3", style={'color': COLORS['primary'], 'fontSize': '16px'}),
                
                html.P("¡Excelente! El perfil analizado presenta una alta probabilidad de suscripción.", 
                       className="mb-3", style={'color': COLORS['text']}),
                html.P("Acciones recomendadas:", 
                       className="mb-2 fw-semibold", style={'color': COLORS['text']}),
                html.Ul([
                    html.Li("Contactar al cliente de inmediato", className="mb-2"),
                    html.Li("Preparar oferta personalizada basada en su perfil", className="mb-2"),
                    html.Li("Asignar a un agente senior para el seguimiento", className="mb-2"),
                    html.Li("Establecer un plan de seguimiento de 3-5 días"),
                ], style={'color': COLORS['text_light'], 'lineHeight': '1.8'})
            ]
            
        else:
            prob_texto = "Baja"
            prob_desc = "Probabilidad de Suscripción"
            badge_text = "✗ Suscripción Poco Probable"
            badge_class = "badge bg-danger mt-2"
            icono_class = "bi bi-x-circle-fill"
            icono_style = {'fontSize': '40px', 'color': '#ef4444'}
            
            recomendaciones = [
                html.Div([
                    html.I(className="bi bi-lightbulb-fill me-2", style={'color': COLORS['accent']}),
                    html.Span("Recomendaciones Estratégicas", className="fw-semibold")
                ], className="d-flex align-items-center mb-3", style={'color': COLORS['primary'], 'fontSize': '16px'}),
                
                html.P("El perfil analizado presenta una baja probabilidad de suscripción.", 
                       className="mb-3", style={'color': COLORS['text']}),
                html.P("Considere las siguientes estrategias:", 
                       className="mb-2 fw-semibold", style={'color': COLORS['text']}),
                html.Ul([
                    html.Li("Reevaluar el momento del contacto", className="mb-2"),
                    html.Li("Ajustar la oferta a las necesidades del cliente", className="mb-2"),
                    html.Li("Considerar contactar en un mes diferente", className="mb-2"),
                    html.Li("Reducir la frecuencia de contactos para evitar saturación"),
                ], style={'color': COLORS['text_light'], 'lineHeight': '1.8'})
            ]
        
        return (
            prob_texto,
            prob_desc,
            html.Span(badge_text, className=badge_class, style={'fontSize': '12px'}),
            icono_class,
            icono_style,
            recomendaciones
        )
        
    except requests.exceptions.Timeout:
        return (
            "Error",
            "Tiempo de espera agotado",
            html.Span("⚠ Timeout", className="badge bg-warning mt-2", style={'fontSize': '12px'}),
            "bi bi-exclamation-triangle-fill",
            {'fontSize': '40px', 'color': '#f59e0b'},
            [html.P("La API tardó demasiado en responder. Intente nuevamente.", 
                   style={'color': COLORS['text']})]
        )
    except requests.exceptions.RequestException as e:
        return (
            "Error",
            "Error de conexión con la API",
            html.Span("⚠ Error", className="badge bg-danger mt-2", style={'fontSize': '12px'}),
            "bi bi-exclamation-triangle-fill",
            {'fontSize': '40px', 'color': '#ef4444'},
            [html.P(f"No se pudo conectar con la API: {str(e)}", 
                   style={'color': COLORS['text']})]
        )
    except Exception as e:
        return (
            "Error",
            "Error inesperado",
            html.Span("⚠ Error", className="badge bg-danger mt-2", style={'fontSize': '12px'}),
            "bi bi-exclamation-triangle-fill",
            {'fontSize': '40px', 'color': '#ef4444'},
            [html.P(f"Error al procesar: {str(e)}", 
                   style={'color': COLORS['text']})]
        )


server = app.server  

if __name__ == "__main__":
    app.run(debug=True)