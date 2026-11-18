# {{ report.title|md }}

{% for line in report.meta %}
- {{ line|md }}
{% endfor %}
{% for table in report.tables %}

{{ "#" * table.level }} {{ table.heading|md }}
{% if table.intro %}

{{ table.intro|md }}
{% endif %}
{% if table.rows %}

|{% for h in table.header %} {{ h|md }} |{% endfor %}

|{% for h in table.header %}---|{% endfor %}

{% for row in table.rows %}
|{% for cell in row %} {{ cell.text|md }} |{% endfor %}

{% endfor %}
{% endif %}
{% if table.notes %}

{% for note in table.notes %}
- {{ note|md }}
{% endfor %}
{% endif %}
{% endfor %}
