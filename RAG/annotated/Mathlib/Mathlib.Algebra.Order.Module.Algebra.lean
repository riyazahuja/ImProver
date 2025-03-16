@[mono] lemma algebraMap_mono : Monotone (algebraMap α β) :=
  fun a₁ a₂ ha ↦ by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : OrderedCommSemiring α
      inst✝² : OrderedSemiring β
      inst✝¹ : Algebra α β
      inst✝ : SMulPosMono α β
      a₁ a₂ : α
      ha : LE.le a₁ a₂
      ⊢ LE.le ((algebraMap α β) a₁) ((algebraMap α β) a₂)
    -/
    simpa only [Algebra.algebraMap_eq_smul_one] using smul_le_smul_of_nonneg_right ha zero_le_one
    /-
      🎉 no goals
    -/


/-- A version of `algebraMap_mono` for use by `gcongr` since it currently does not preprocess
`Monotone` conclusions. -/
@[gcongr] protected lemma GCongr.algebraMap_le_algebraMap {a₁ a₂ : α} (ha : a₁ ≤ a₂) :
    algebraMap α β a₁ ≤ algebraMap α β a₂ := algebraMap_mono _ ha


                                                                  /-
                                                                    α : Type u_1
                                                                    β : Type u_2
                                                                    inst✝³ : OrderedCommSemiring α
                                                                    inst✝² : OrderedSemiring β
                                                                    inst✝¹ : Algebra α β
                                                                    inst✝ : SMulPosMono α β
                                                                    a : α
                                                                    ha : LE.le 0 a
                                                                    ⊢ LE.le 0 ((algebraMap α β) a)
                                                                  -/
lemma algebraMap_nonneg (ha : 0 ≤ a) : 0 ≤ algebraMap α β a := by simpa using algebraMap_mono β ha
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp] lemma algebraMap_le_algebraMap : algebraMap α β a₁ ≤ algebraMap α β a₂ ↔ a₁ ≤ a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : OrderedCommSemiring α
    inst✝³ : StrictOrderedSemiring β
    inst✝² : Algebra α β
    inst✝¹ : SMulPosMono α β
    inst✝ : SMulPosReflectLE α β
    a₁ a₂ : α
    ⊢ Iff (LE.le ((algebraMap α β) a₁) ((algebraMap α β) a₂)) (LE.le a₁ a₂)
  -/
  simp [Algebra.algebraMap_eq_smul_one]
  /-
    🎉 no goals
  -/


@[mono] lemma algebraMap_strictMono : StrictMono (algebraMap α β) :=
  fun a₁ a₂ ha ↦ by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : OrderedCommSemiring α
      inst✝² : StrictOrderedSemiring β
      inst✝¹ : Algebra α β
      inst✝ : SMulPosStrictMono α β
      a₁ a₂ : α
      ha : LT.lt a₁ a₂
      ⊢ LT.lt ((algebraMap α β) a₁) ((algebraMap α β) a₂)
    -/
    simpa only [Algebra.algebraMap_eq_smul_one] using smul_lt_smul_of_pos_right ha zero_lt_one
    /-
      🎉 no goals
    -/


/-- A version of `algebraMap_strictMono` for use by `gcongr` since it currently does not preprocess
`Monotone` conclusions. -/
@[gcongr] protected lemma GCongr.algebraMap_lt_algebraMap {a₁ a₂ : α} (ha : a₁ < a₂) :
    algebraMap α β a₁ < algebraMap α β a₂ := algebraMap_strictMono _ ha


lemma algebraMap_pos (ha : 0 < a) : 0 < algebraMap α β a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : OrderedCommSemiring α
    inst✝² : StrictOrderedSemiring β
    inst✝¹ : Algebra α β
    inst✝ : SMulPosStrictMono α β
    a : α
    ha : LT.lt 0 a
    ⊢ LT.lt 0 ((algebraMap α β) a)
  -/
  simpa using algebraMap_strictMono β ha
  /-
    🎉 no goals
  -/


@[simp] lemma algebraMap_lt_algebraMap : algebraMap α β a₁ < algebraMap α β a₂ ↔ a₁ < a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : OrderedCommSemiring α
    inst✝³ : StrictOrderedSemiring β
    inst✝² : Algebra α β
    inst✝¹ : SMulPosStrictMono α β
    a₁ a₂ : α
    inst✝ : SMulPosReflectLT α β
    ⊢ Iff (LT.lt ((algebraMap α β) a₁) ((algebraMap α β) a₂)) (LT.lt a₁ a₂)
  -/
  simp [Algebra.algebraMap_eq_smul_one]
  /-
    🎉 no goals
  -/


/-- Extension for `algebraMap`. -/
@[positivity algebraMap _ _ _]
def evalAlgebraMap : PositivityExt where eval {u β} _zβ _pβ e := do
  let ~q(@algebraMap $α _ $instα $instβ $instαβ $a) := e | throwError "not `algebraMap`"
  let pα ← synthInstanceQ (q(PartialOrder $α) : Q(Type u_1))
  match ← core q(inferInstance) pα a with
  | .positive pa =>
    let _instαring ← synthInstanceQ q(OrderedCommSemiring $α)
    try
      let _instβring ← synthInstanceQ q(StrictOrderedSemiring $β)
      let _instαβsmul ← synthInstanceQ q(SMulPosStrictMono $α $β)
      assertInstancesCommute
      return .positive q(algebraMap_pos $β $pa)
    catch _ =>
      let _instβring ← synthInstanceQ q(OrderedSemiring $β)
      let _instαβsmul ← synthInstanceQ q(SMulPosMono $α $β)
      assertInstancesCommute
      return .nonnegative q(algebraMap_nonneg $β <| le_of_lt $pa)
  | .nonnegative pa =>
    let _instαring ← synthInstanceQ q(OrderedCommSemiring $α)
    let _instβring ← synthInstanceQ q(OrderedSemiring $β)
    let _instαβsmul ← synthInstanceQ q(SMulPosMono $α $β)
    assertInstancesCommute
    return .nonnegative q(algebraMap_nonneg $β $pa)
  | _ => pure .none


