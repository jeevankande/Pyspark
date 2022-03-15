
#unpivot data in pyspark
from pyspark.sql.functions import array, col, explode, lit, struct
from pyspark.sql import DataFrame
from typing import Iterable

def melt_df(
        df: DataFrame,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str="variable", value_name: str="value"):
    """Convert :class:`DataFrame` from wide to long format."""

    # Create array
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name))
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))

    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)



# column headers
table_columns = ["002_2022","003_2022","004_2022","005_2022","006_2022","007_2022","008_2022","009_2022","010_2022","011_2022","012_2022","001_2023","002_2023"]



df_rel = melt_df(sparkDF, ['PLANT','FRANCHISE','Product'], table_columns, 'Month', 'Forecast')



df_rel.show()

