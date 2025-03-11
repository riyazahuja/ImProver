theorem isLinearMap_add [AddCommMonoid M] [Module R M] :
    IsLinearMap R fun x : M × M => x.1 + x.2 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ IsLinearMap R fun x => HAdd.hAdd x.1 x.2
  -/
  apply IsLinearMap.mk
    /-
      case map_add
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ⊢ ∀ (x y : Prod M M), Eq (HAdd.hAdd (HAdd.hAdd x y).1 (HAdd.hAdd x y).2) (HAdd …
    -/
  · intro x y
    /-
      case map_add
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x y : Prod M M
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd x y).1 (HAdd.hAdd x y).2) (HAdd.hAdd (HAdd.hAdd x.1 …
    -/
    simp only [Prod.fst_add, Prod.snd_add]
    /-
      case map_add
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x y : Prod M M
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd x.1 y.1) (HAdd.hAdd x.2 y.2)) (HAdd.hAdd (HAdd.hAdd …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
    /-
      case map_smul
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ⊢ ∀ (c : R) (x : Prod M M), Eq (HAdd.hAdd (HSMul.hSMul c x).1 (HSMul.hSMul c x …
    -/
  · intro x y
    /-
      case map_smul
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x : R
      y : Prod M M
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul x y).1 (HSMul.hSMul x y).2) (HSMul.hSMul x (HAdd. …
    -/
    simp [smul_add]
    /-
      🎉 no goals
    -/


theorem isLinearMap_sub [AddCommGroup M] [Module R M] :
    IsLinearMap R fun x : M × M => x.1 - x.2 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ IsLinearMap R fun x => HSub.hSub x.1 x.2
  -/
  apply IsLinearMap.mk
    /-
      case map_add
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      ⊢ ∀ (x y : Prod M M), Eq (HSub.hSub (HAdd.hAdd x y).1 (HAdd.hAdd x y).2) (HAdd …
    -/
  · intro x y
    /-
      case map_add
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : Prod M M
      ⊢ Eq (HSub.hSub (HAdd.hAdd x y).1 (HAdd.hAdd x y).2) (HAdd.hAdd (HSub.hSub x.1 …
    -/
    simp [add_comm, add_assoc, add_left_comm, sub_eq_add_neg]
    /-
      🎉 no goals
    -/
    /-
      case map_smul
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      ⊢ ∀ (c : R) (x : Prod M M), Eq (HSub.hSub (HSMul.hSMul c x).1 (HSMul.hSMul c x …
    -/
  · intro x y
    /-
      case map_smul
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : R
      y : Prod M M
      ⊢ Eq (HSub.hSub (HSMul.hSMul x y).1 (HSMul.hSMul x y).2) (HSMul.hSMul x (HSub. …
    -/
    simp [smul_sub]
    /-
      🎉 no goals
    -/


