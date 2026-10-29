import pytest

from prada import cli, persona


def test_default_persona_has_the_guardrail():
    assert persona.CURRENT == persona.Persona()
    assert persona.CHARACTER.startswith("You are PRADA, ")
    assert persona.CHARACTER.endswith(persona.GUARDRAIL)
    assert persona.system_prompt("Task.").endswith("\n\nTask.")


def test_missing_file_is_the_default(tmp_path):
    assert persona.load(tmp_path / "persona.toml") == persona.Persona()


def test_file_sets_name_character_and_tone(tmp_path):
    path = tmp_path / "persona.toml"
    path.write_text(
        'name = "Aiko"\ncharacter = """\nYou are {name}, cheerful."""\ntone = "bright"\n'
    )
    loaded = persona.load(path)
    assert loaded == persona.Persona("Aiko", "You are {name}, cheerful.", "bright")
    assert persona.render(loaded) == f"You are Aiko, cheerful. {persona.GUARDRAIL}"


def test_missing_keys_keep_the_default(tmp_path):
    path = tmp_path / "persona.toml"
    path.write_text('tone = "dry"\n')
    assert persona.load(path) == persona.Persona(tone="dry")


@pytest.mark.parametrize(
    "text, error",
    [
        ('nme = "x"\n', "unknown keys"),
        ('name = ""\n', "non-empty string"),
        ("tone = 3\n", "non-empty string"),
        ("name = \n", "can't read"),
    ],
)
def test_bad_files_are_errors(tmp_path, text, error):
    path = tmp_path / "persona.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        persona.load(path)


@pytest.mark.parametrize(
    "character",
    ["Plain.", 'She says "hi" \\o/', "Line one.\nLine two.", "Has ''' quotes and \"\"\" too."],
)
def test_template_round_trips(tmp_path, character):
    path = tmp_path / "persona.toml"
    wanted = persona.Persona('Kai "K"', character, "warm")
    path.write_text(persona.template(wanted))
    assert persona.load(path) == wanted


def test_env_var_picks_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PRADA_PERSONA", str(tmp_path / "p.toml"))
    assert persona.persona_path() == tmp_path / "p.toml"
    monkeypatch.delenv("PRADA_PERSONA")
    assert persona.persona_path() == tmp_path / "persona.toml"  # next to PRADA_CONFIG


def test_cli_init_writes_once_and_shows_it(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("PRADA_PERSONA")
    path = tmp_path / "persona.toml"
    assert cli.main(["persona"]) == 0
    assert "the default" in capsys.readouterr().out
    assert cli.main(["persona", "init"]) == 0
    assert persona.load(path) == persona.CURRENT
    assert cli.main(["persona", "init"]) == 1
    assert "already exists" in capsys.readouterr().err
    path.write_text('name = "Aiko"\n')
    assert cli.main(["persona"]) == 0
    assert "You are Aiko, " in capsys.readouterr().out
    assert cli.main(["persona", "init", "--force"]) == 0
    assert persona.load(path) == persona.CURRENT


def test_cli_reports_a_broken_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_PERSONA", str(tmp_path / "p.toml"))
    (tmp_path / "p.toml").write_text("oops = 1\n")
    assert cli.main(["persona"]) == 1
    assert "unknown keys" in capsys.readouterr().err
