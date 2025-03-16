theorem _root_.IsSMulRegular.pi {α : Type*} [∀ i, SMul α <| f i] {k : α}
    (hk : ∀ i, IsSMulRegular (f i) k) : IsSMulRegular (∀ i, f i) k := fun _ _ h =>
  funext fun i => hk i (congr_fun h i : _)


instance smulWithZero (α) [Zero α] [∀ i, Zero (f i)] [∀ i, SMulWithZero α (f i)] :
    SMulWithZero α (∀ i, f i) :=
  { Pi.instSMul with
    smul_zero := fun _ => funext fun _ => smul_zero _
    zero_smul := fun _ => funext fun _ => zero_smul _ _ }


instance smulWithZero' {g : I → Type*} [∀ i, Zero (g i)] [∀ i, Zero (f i)]
    [∀ i, SMulWithZero (g i) (f i)] : SMulWithZero (∀ i, g i) (∀ i, f i) :=
  { Pi.smul' with
    smul_zero := fun _ => funext fun _ => smul_zero _
    zero_smul := fun _ => funext fun _ => zero_smul _ _ }


instance mulActionWithZero (α) [MonoidWithZero α] [∀ i, Zero (f i)]
    [∀ i, MulActionWithZero α (f i)] : MulActionWithZero α (∀ i, f i) :=
  { Pi.mulAction _, Pi.smulWithZero _ with }


instance mulActionWithZero' {g : I → Type*} [∀ i, MonoidWithZero (g i)] [∀ i, Zero (f i)]
    [∀ i, MulActionWithZero (g i) (f i)] : MulActionWithZero (∀ i, g i) (∀ i, f i) :=
  { Pi.mulAction', Pi.smulWithZero' with }


instance module (α) {r : Semiring α} {m : ∀ i, AddCommMonoid <| f i} [∀ i, Module α <| f i] :
    @Module α (∀ i : I, f i) r (@Pi.addCommMonoid I f m) :=
  { Pi.distribMulAction _ with
    add_smul := fun _ _ _ => funext fun _ => add_smul _ _ _
    zero_smul := fun _ => funext fun _ => zero_smul α _ }

/- Extra instance to short-circuit type class resolution.
For unknown reasons, this is necessary for certain inference problems. E.g., for this to succeed:
```lean
example (β X : Type*) [NormedAddCommGroup β] [NormedSpace ℝ β] : Module ℝ (X → β) := inferInstance
```
See: https://leanprover.zulipchat.com/#narrow/stream/113488-general/topic/Typeclass.20resolution.20under.20binders/near/281296989
-/

/-- A special case of `Pi.module` for non-dependent types. Lean struggles to elaborate
definitions elsewhere in the library without this. -/
instance Function.module (α β : Type*) [Semiring α] [AddCommMonoid β] [Module α β] :
    Module α (I → β) :=
  Pi.module _ _ _


instance module' {g : I → Type*} {r : ∀ i, Semiring (f i)} {m : ∀ i, AddCommMonoid (g i)}
    [∀ i, Module (f i) (g i)] : Module (∀ i, f i) (∀ i, g i) where
  add_smul := by
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      r : (i : I) → Semiring (f i)
      m : (i : I) → AddCommMonoid (g i)
      inst✝ : (i : I) → Module (f i) (g i)
      ⊢ ∀ (r_1 s : (i : I) → f i) (x : (i : I) → g i), Eq (HSMul.hSMul (HAdd.hAdd r_ …
    -/
    intros
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      r : (i : I) → Semiring (f i)
      m : (i : I) → AddCommMonoid (g i)
      inst✝ : (i : I) → Module (f i) (g i)
      r✝ s✝ : (i : I) → f i
      x✝ : (i : I) → g i
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd r✝ s✝) x✝) (HAdd.hAdd (HSMul.hSMul r✝ x✝) (HSMul. …
    -/
    ext1
    /-
      case h
      I : Type u
      f : I → Type v
      g : I → Type u_1
      r : (i : I) → Semiring (f i)
      m : (i : I) → AddCommMonoid (g i)
      inst✝ : (i : I) → Module (f i) (g i)
      r✝ s✝ : (i : I) → f i
      x✝¹ : (i : I) → g i
      x✝ : I
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd r✝ s✝) x✝¹ x✝) (HAdd.hAdd (HSMul.hSMul r✝ x✝¹) (H …
    -/
    apply add_smul
    /-
      🎉 no goals
    -/
  zero_smul := by
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      r : (i : I) → Semiring (f i)
      m : (i : I) → AddCommMonoid (g i)
      inst✝ : (i : I) → Module (f i) (g i)
      ⊢ ∀ (x : (i : I) → g i), Eq (HSMul.hSMul 0 x) 0
    -/
    intros
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      r : (i : I) → Semiring (f i)
      m : (i : I) → AddCommMonoid (g i)
      inst✝ : (i : I) → Module (f i) (g i)
      x✝ : (i : I) → g i
      ⊢ Eq (HSMul.hSMul 0 x✝) 0
    -/
    ext1
    -- Porting note: not sure why `apply zero_smul` fails here.
    /-
      case h
      I : Type u
      f : I → Type v
      g : I → Type u_1
      r : (i : I) → Semiring (f i)
      m : (i : I) → AddCommMonoid (g i)
      inst✝ : (i : I) → Module (f i) (g i)
      x✝¹ : (i : I) → g i
      x✝ : I
      ⊢ Eq (HSMul.hSMul 0 x✝¹ x✝) (0 x✝)
    -/
    rw [zero_smul]
    /-
      🎉 no goals
    -/


