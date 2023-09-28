using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MauiApp2.Data
{
    public class AnnotatedExample
    {
        // id,text,subtask_a,subtask_b,subtask_c

        [LoadColumn(0)]
        public int Id { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }

        [LoadColumn(2)]
        public string SubtaskA { get; set; }

        [LoadColumn(3)]
        public string SubtaskB { get; set; }

        [LoadColumn(4)]
        public string SubtaskC { get; set; }
    }
}
