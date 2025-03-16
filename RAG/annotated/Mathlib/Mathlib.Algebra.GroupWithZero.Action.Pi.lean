instance smulZeroClass (α) {n : ∀ i, Zero <| f i} [∀ i, SMulZeroClass α <| f i] :
  @SMulZeroClass α (∀ i : I, f i) (@Pi.instZero I f n) where
  smul_zero _ := funext fun _ => smul_zero _


instance smulZeroClass' {g : I → Type*} {n : ∀ i, Zero <| g i} [∀ i, SMulZeroClass (f i) (g i)] :
  @SMulZeroClass (∀ i, f i) (∀ i : I, g i) (@Pi.instZero I g n) where
                  /-
                    I : Type u
                    f : I → Type v
                    g : I → Type u_1
                    n : (i : I) → Zero (g i)
                    inst✝ : (i : I) → SMulZeroClass (f i) (g i)
                    ⊢ ∀ (a : (i : I) → f i), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_zero := by intros; ext x; exact smul_zero _
                                 /-
                                   🎉 no goals
                                 -/


instance distribSMul (α) {n : ∀ i, AddZeroClass <| f i} [∀ i, DistribSMul α <| f i] :
  @DistribSMul α (∀ i : I, f i) (@Pi.addZeroClass I f n) where
  smul_zero _ := funext fun _ => smul_zero _
  smul_add _ _ _ := funext fun _ => smul_add _ _ _


instance distribSMul' {g : I → Type*} {n : ∀ i, AddZeroClass <| g i}
  [∀ i, DistribSMul (f i) (g i)] :
  @DistribSMul (∀ i, f i) (∀ i : I, g i) (@Pi.addZeroClass I g n) where
                  /-
                    I : Type u
                    f : I → Type v
                    g : I → Type u_1
                    n : (i : I) → AddZeroClass (g i)
                    inst✝ : (i : I) → DistribSMul (f i) (g i)
                    ⊢ ∀ (a : (i : I) → f i), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_zero := by intros; ext x; exact smul_zero _
                                 /-
                                   🎉 no goals
                                 -/
                 /-
                   I : Type u
                   f : I → Type v
                   g : I → Type u_1
                   n : (i : I) → AddZeroClass (g i)
                   inst✝ : (i : I) → DistribSMul (f i) (g i)
                   ⊢ ∀ (a : (i : I) → f i) (x y : (i : I) → g i), Eq (HSMul.hSMul a (HAdd.hAdd x  …
                 -/
  smul_add := by intros; ext x; exact smul_add _ _ _
                                /-
                                  🎉 no goals
                                -/


instance distribMulAction (α) {m : Monoid α} {n : ∀ i, AddMonoid <| f i}
    [∀ i, DistribMulAction α <| f i] : @DistribMulAction α (∀ i : I, f i) m (@Pi.addMonoid I f n) :=
  { Pi.mulAction _, Pi.distribSMul _ with }


instance distribMulAction' {g : I → Type*} {m : ∀ i, Monoid (f i)} {n : ∀ i, AddMonoid <| g i}
    [∀ i, DistribMulAction (f i) (g i)] :
    @DistribMulAction (∀ i, f i) (∀ i : I, g i) (@Pi.monoid I f m) (@Pi.addMonoid I g n) :=
  { Pi.mulAction', Pi.distribSMul' with }


theorem single_smul {α} [Monoid α] [∀ i, AddMonoid <| f i] [∀ i, DistribMulAction α <| f i]
    [DecidableEq I] (i : I) (r : α) (x : f i) : single i (r • x) = r • single i x :=
  single_op (fun i : I => (r • · : f i → f i)) (fun _ => smul_zero _) _ _

-- Porting note: Lean4 cannot infer the non-dependent function `f := fun _ => β`

/-- A version of `Pi.single_smul` for non-dependent functions. It is useful in cases where Lean
fails to apply `Pi.single_smul`. -/
theorem single_smul' {α β} [Monoid α] [AddMonoid β] [DistribMulAction α β] [DecidableEq I] (i : I)
    (r : α) (x : β) : single (f := fun _ => β) i (r • x) = r • single (f := fun _ => β) i x :=
  single_smul (f := fun _ => β) i r x


theorem single_smul₀ {g : I → Type*} [∀ i, MonoidWithZero (f i)] [∀ i, AddMonoid (g i)]
    [∀ i, DistribMulAction (f i) (g i)] [DecidableEq I] (i : I) (r : f i) (x : g i) :
    single i (r • x) = single i r • single i x :=
  single_op₂ (fun i : I => ((· • ·) : f i → g i → g i)) (fun _ => smul_zero _) _ _ _


instance mulDistribMulAction (α) {m : Monoid α} {n : ∀ i, Monoid <| f i}
    [∀ i, MulDistribMulAction α <| f i] :
    @MulDistribMulAction α (∀ i : I, f i) m (@Pi.monoid I f n) :=
  { Pi.mulAction _ with
    smul_one := fun _ => funext fun _ => smul_one _
    smul_mul := fun _ _ _ => funext fun _ => smul_mul' _ _ _ }


instance mulDistribMulAction' {g : I → Type*} {m : ∀ i, Monoid (f i)} {n : ∀ i, Monoid <| g i}
    [∀ i, MulDistribMulAction (f i) (g i)] :
    @MulDistribMulAction (∀ i, f i) (∀ i : I, g i) (@Pi.monoid I f m) (@Pi.monoid I g n) where
  smul_mul := by
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      m : (i : I) → Monoid (f i)
      n : (i : I) → Monoid (g i)
      inst✝ : (i : I) → MulDistribMulAction (f i) (g i)
      ⊢ ∀ (r : (i : I) → f i) (x y : (i : I) → g i), Eq (HSMul.hSMul r (HMul.hMul x  …
    -/
    intros
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      m : (i : I) → Monoid (f i)
      n : (i : I) → Monoid (g i)
      inst✝ : (i : I) → MulDistribMulAction (f i) (g i)
      r✝ : (i : I) → f i
      x✝ y✝ : (i : I) → g i
      ⊢ Eq (HSMul.hSMul r✝ (HMul.hMul x✝ y✝)) (HMul.hMul (HSMul.hSMul r✝ x✝) (HSMul. …
    -/
    ext x
    /-
      case h
      I : Type u
      f : I → Type v
      g : I → Type u_1
      m : (i : I) → Monoid (f i)
      n : (i : I) → Monoid (g i)
      inst✝ : (i : I) → MulDistribMulAction (f i) (g i)
      r✝ : (i : I) → f i
      x✝ y✝ : (i : I) → g i
      x : I
      ⊢ Eq (HSMul.hSMul r✝ (HMul.hMul x✝ y✝) x) (HMul.hMul (HSMul.hSMul r✝ x✝) (HSMu …
    -/
    apply smul_mul'
    /-
      🎉 no goals
    -/
  smul_one := by
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      m : (i : I) → Monoid (f i)
      n : (i : I) → Monoid (g i)
      inst✝ : (i : I) → MulDistribMulAction (f i) (g i)
      ⊢ ∀ (r : (i : I) → f i), Eq (HSMul.hSMul r 1) 1
    -/
    intros
    /-
      I : Type u
      f : I → Type v
      g : I → Type u_1
      m : (i : I) → Monoid (f i)
      n : (i : I) → Monoid (g i)
      inst✝ : (i : I) → MulDistribMulAction (f i) (g i)
      r✝ : (i : I) → f i
      ⊢ Eq (HSMul.hSMul r✝ 1) 1
    -/
    ext x
    /-
      case h
      I : Type u
      f : I → Type v
      g : I → Type u_1
      m : (i : I) → Monoid (f i)
      n : (i : I) → Monoid (g i)
      inst✝ : (i : I) → MulDistribMulAction (f i) (g i)
      r✝ : (i : I) → f i
      x : I
      ⊢ Eq (HSMul.hSMul r✝ 1 x) (1 x)
    -/
    apply smul_one
    /-
      🎉 no goals
    -/


