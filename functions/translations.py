# Column → label in English
VARIABLE_LABELS = {
    "delito1_descripcion": "Main offense",
    "nivel_instruccion_descripcion": "Education level",
    "ultima_situacion_laboral_descripcion": "Previous employment status",
    "calificacion_conducta_descripcion": "Institutional conduct rating",
    "ultima_provincia_residencia_descripcion": "Province of last residence",
    "tiene_periodo_progresividad_descripcion": "Progression in regime",
    "tipo_infraccion_disciplinaria_descripcion": "Disciplinary infractions",
    "participa_programa_pre_libertad": "Pre-release program participation",
    "participacion_actividades_deportivas": "Sports activities participation",
    "genero_descripcion": "Gender",
    "edad": "Age",
    "duracion_condena_anios": "Sentence length (years)",
    "es_reincidente_descripcion": "Recidivism status",
}
TARGET_LABELS_EN = {
    "Primario/a": "Primary",
    "Reincidente": "Recidivist",
    "Reiterante": "Repeat offender",
}
# Column → {category_in_spanish: category_in_english}
CATEGORY_LABELS = {
    "es_reincidente_descripcion": {
        "Primario/a": "First-time offender",
        "Reiterante": "Repeat offender",
        "Reincidente (art. 50 CP)": "Reoffender",
    },
    "calificacion_conducta_descripcion": {
        "Ejemplar": "Exemplary",
        "Buena": "Good",
        "Muy buena": "Very good",
        "Sin calificación": "Not rated",
        "Regular": "Fair",
        "Mala": "Poor",
        "Pésima": "Very poor",
    },

    "ultima_situacion_laboral_descripcion": {
        "Trabajador/ra de tiempo parcial": "Part-time worker",
        "Desocupado/a": "Unemployed",
        "Trabajador/ra de tiempo completo": "Full-time worker",
        "Desconocido": "Unknown",
    },

    "nivel_instruccion_descripcion": {
        "Primario completo": "Primary complete",
        "Secundario incompleto": "Secondary incomplete",
        "Primario incompleto": "Primary incomplete",
        "Secundario completo": "Secondary complete",
        "Ninguno": "No formal education",
        "Terciario completo": "Tertiary complete",
        "Universitario incompleto": "University incomplete",
        "Terciario incompleto": "Tertiary incomplete",
        "Desconocido": "Unknown",
        "Universitario completo": "University complete",
    },

    "tipo_infraccion_disciplinaria_descripcion": {
        "No cometió Infracción disciplinaria": "No disciplinary infraction",
        "Faltas graves": "Serious offenses",
        "Faltas media": "Medium offenses",
        "Faltas leves": "Minor offenses",
    },

    "delito1_descripcion": {
        "Robo y/o tentativa de robo": "Robbery and/or attempted robbery",
        "Violaciones/Abuso sexual": "Rape / Sexual abuse",
        "Homicidios dolosos": "Intentional homicide",
        "Infracción ley n° 23.737 (estupefacientes)": "Drug offenses (Law 23.737)",
        "Lesiones Dolosas": "Intentional injuries",
        "Hurto y/o tentativa de hurto": "Theft and/or attempted theft",
        "Amenazas": "Threats",
        "Homicidios Culposos": "Unintentional homicide",
        "Otros delitos contra la propiedad": "Other property offenses",
        "Homicidios dolosos (tent.)": "Attempted intentional homicide",
        "Otros delitos contra las personas": "Other offenses against persons",
        "Delitos c/ la administracion pública": "Offenses against public administration",
        "Delitos contra la seguridad pública": "Offenses against public safety",
        "Otros delitos contra la libertad": "Other offenses against personal freedom",
        "Otros delitos contra la integridad sexual": "Other sexual integrity offenses",
        "Privación ilegítima de la libertad": "Unlawful deprivation of liberty",
        "Delitos c/el orden público": "Offenses against public order",
        "Delitos contra el honor": "Offenses against honor",
        "Lesiones Culposas": "Unintentional injuries",
        "Delitos c/ la fe pública": "Offenses against public faith",
        "Contrabando de estupefacientes": "Drug smuggling",
        "Delito no especificado": "Unspecified offense",
        "Delitos previstos en leyes especiales": "Offenses under special laws",
        "Lesa Humanidad": "Crimes against humanity",
        "Del. contra la lib. comet. por func. público": "Offenses against personal freedom committed by public officials",
        "Infraccion ley n° 24.769 penal tributaria": "Tax offenses (Law 24.769)",
        "Delitos c/ el estado civil": "Offenses against civil status",
        "Delitos contra los poderes publicos": "Offenses against public authorities",
        "Contravenciones": "Misdemeanors",
        "Infraccion ley n° 13.944 incumplimiento de deberes": "Failure to fulfill duties (Law 13.944)",
        "Delitos contra la seguridad de la nación": "Offenses against national security",
    },

    "tiene_periodo_progresividad_descripcion": {
        "Período de tratamiento": "Treatment phase",
        "No se aplica ninguna": "No progression phase applied",
        "Período de prueba": "Probation phase",
        "Período de observación": "Observation phase",
        "Se aplica otra caracterización": "Other classification applied",
        "Período de libertad condicional": "Parole phase",
    },

    "genero_descripcion": {
        "Varón": "Male",
        "Mujer": "Female",
    }
}

