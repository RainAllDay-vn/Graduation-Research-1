# Detailed LC-QuAD 2.0 Template Matching

This table maps each of the 26 unique templates found in the `train_templates_info.json` to the corresponding research type from the dataset report.

| # | Template String | Frequency | Matching Research Type | Reason/Description |
| :-- | :--- | :--- | :--- | :--- |
| 1 | `E REF ?F` | 2,588 | **Type 1: Single fact** | Direct subject-predicate-object query for object. |
| 2 | `(E pred F) prop ?value` | 2,320 | **Type 4: Fact with qualifiers** | Query for a qualifier value of a main triple. |
| 3 | `(E pred ?Obj ) prop value` | 2,301 | **Type 4: Fact with qualifiers** | Query for an object given its qualifier value. |
| 4 | `E REF ?F . ?F RFG G` | 1,996 | **Type 3: Multi-fact** | 2-hop path query (joined by ?F). |
| 5 | `<?S P O ; ?S InstanceOf Type>` | 1,553 | **Type 2: Single fact with type** | Single triple query with a subject type constraint. |
| 6 | `E REF xF . xF RFG ?G` | 1,516 | **Type 3: Multi-fact** | 2-hop path where object of second hop is the goal. |
| 7 | `<S P ?O ; ?O instanceOf Type>` | 1,434 | **Type 2: Single fact with type** | Single triple query with an object type constraint. |
| 8 | `C RCD xD . xD RDE ?E` | 1,363 | **Type 3: Multi-fact** | Another variation of a joined multi-fact query. |
| 9 | `ASK ?sbj ?pred ?obj filter ?obj = num` | 1,302 | **Type 6: Boolean** | Boolean question with numerical comparison. |
| 10 | `[]` | 1,217 | **N/A** | Empty/Incomplete entry found in dataset logs. |
| 11 | `<?S P O ; ?S instanceOf Type ; starts with character >` | 1,028 | **Type 9: String Operation** | Character-level filtering (starts with). |
| 12 | `<?S P O ; ?S instanceOf Type ; contains word >` | 1,020 | **Type 9: String Operation** | Word-level filtering (contains). |
| 13 | `Count ent (ent-pred-obj)` | 610 | **Type 7: Count** | Returns number of matching subjects. |
| 14 | `select where (ent-pred-obj1 . ent-pred-obj2)` | 584 | **Type 5: Two intention** | Multiple intentions expressed via separate triple patterns. |
| 15 | `?D RDE E` | 572 | **Type 1: Single fact** | Direct subject-predicate-object query for subject. |
| 16 | `Count Obj (ent-pred-obj)` | 501 | **Type 7: Count** | Returns number of matching objects. |
| 17 | `?E is_a Type, ?E pred Obj value. MAX/MIN (value)` | 298 | **Type 8: Ranking** | Finds entity with max/min value within a type context. |
| 18 | `?E is_a Type. ?E pred Obj. ?E-secondClause value. MIN (value)`| 294 | **Type 8: Ranking** | Multi-hop ranking for minimum value. |
| 19 | `?E is_a Type. ?E pred Obj. ?E-secondClause value. MAX (value)` | 288 | **Type 8: Ranking** | Multi-hop ranking for maximum value. |
| 20 | `Ask (ent-pred-obj)` | 255 | **Type 6: Boolean** | Simple True/False fact checking. |
| 21 | `Ask (ent-pred-obj1 . ent-pred-obj2)` | 169 | **Type 6: Boolean** | Multi-fact True/False matching. |
| 22 | `Ask (ent-pred-obj\`)` | 133 | **Type 6: Boolean** | Fact checking with qualifier notation. |
| 23 | `Ask (ent-pred-obj1\` . ent-pred-obj2)` | 116 | **Type 6: Boolean** | Boolean query with qualifier and multi-fact. |
| 24 | `Ask (ent-pred-obj1 . ent-pred-obj2\`)` | 100 | **Type 6: Boolean** | Boolean query with qualifier on second fact. |
| 25 | `Ask (ent\`-pred-obj1 . ent\`-pred-obj2)` | 13 | **Type 6: Boolean** | Boolean query with qualifiers on multiple entities. |
| 26 | `Ask (ent\`-pred-obj)` | 6 | **Type 6: Boolean** | Simple boolean check involving qualifiers. |

> [!NOTE]  
> The "Temporal aspect" (Type 10) frequently uses qualifier-based templates (like #2 and #3) to handle date-related properties but is grouped here under the structural type for clarity.
