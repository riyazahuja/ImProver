instance instZero : Zero (OuterMeasure α) :=
  ⟨{  measureOf := fun _ => 0
      empty := rfl
                 /-
                   α : Type u_1
                   β : Type u_2
                   m : MeasureTheory.OuterMeasure α
                   ⊢ ∀ {s₁ s₂ : Set α}, HasSubset.Subset s₁ s₂ → LE.le ((fun x => 0) s₁) ((fun x  …
                 -/
      mono := by intro _ _ _; exact le_refl 0
                              /-
                                🎉 no goals
                              -/
      iUnion_nat := fun _ _ => zero_le _ }⟩


@[simp]
theorem coe_zero : ⇑(0 : OuterMeasure α) = 0 :=
  rfl


instance instInhabited : Inhabited (OuterMeasure α) :=
  ⟨0⟩


instance instAdd : Add (OuterMeasure α) :=
  ⟨fun m₁ m₂ =>
    { measureOf := fun s => m₁ s + m₂ s
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         m m₁ m₂ : MeasureTheory.OuterMeasure α
                                         ⊢ Eq (HAdd.hAdd (m₁ EmptyCollection.emptyCollection) (m₂ EmptyCollection.empty …
                                       -/
      empty := show m₁ ∅ + m₂ ∅ = 0 by simp [OuterMeasure.empty]
                                       /-
                                         🎉 no goals
                                       -/
      mono := fun {_ _} h => add_le_add (m₁.mono h) (m₂.mono h)
      iUnion_nat := fun s _ =>
        calc
          m₁ (⋃ i, s i) + m₂ (⋃ i, s i) ≤ (∑' i, m₁ (s i)) + ∑' i, m₂ (s i) :=
            add_le_add (measure_iUnion_le s) (measure_iUnion_le s)
          _ = _ := ENNReal.tsum_add.symm }⟩


@[simp]
theorem coe_add (m₁ m₂ : OuterMeasure α) : ⇑(m₁ + m₂) = m₁ + m₂ :=
  rfl


theorem add_apply (m₁ m₂ : OuterMeasure α) (s : Set α) : (m₁ + m₂) s = m₁ s + m₂ s :=
  rfl


instance instSMul : SMul R (OuterMeasure α) :=
  ⟨fun c m =>
    { measureOf := fun s => c • m s
                  /-
                    α : Type u_1
                    β : Type u_2
                    m✝ : MeasureTheory.OuterMeasure α
                    R : Type u_3
                    inst✝³ : SMul R ENNReal
                    inst✝² : IsScalarTower R ENNReal ENNReal
                    R' : Type u_4
                    inst✝¹ : SMul R' ENNReal
                    inst✝ : IsScalarTower R' ENNReal ENNReal
                    c : R
                    m : MeasureTheory.OuterMeasure α
                    ⊢ Eq ((fun s => HSMul.hSMul c (m s)) EmptyCollection.emptyCollection) 0
                  -/
      empty := by simp only [measure_empty]; rw [← smul_one_mul c]; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
      mono := fun {s t} h => by
        /-
          α : Type u_1
          β : Type u_2
          m✝ : MeasureTheory.OuterMeasure α
          R : Type u_3
          inst✝³ : SMul R ENNReal
          inst✝² : IsScalarTower R ENNReal ENNReal
          R' : Type u_4
          inst✝¹ : SMul R' ENNReal
          inst✝ : IsScalarTower R' ENNReal ENNReal
          c : R
          m : MeasureTheory.OuterMeasure α
          s t : Set α
          h : HasSubset.Subset s t
          ⊢ LE.le ((fun s => HSMul.hSMul c (m s)) s) ((fun s => HSMul.hSMul c (m s)) t)
        -/
        simp only
        /-
          α : Type u_1
          β : Type u_2
          m✝ : MeasureTheory.OuterMeasure α
          R : Type u_3
          inst✝³ : SMul R ENNReal
          inst✝² : IsScalarTower R ENNReal ENNReal
          R' : Type u_4
          inst✝¹ : SMul R' ENNReal
          inst✝ : IsScalarTower R' ENNReal ENNReal
          c : R
          m : MeasureTheory.OuterMeasure α
          s t : Set α
          h : HasSubset.Subset s t
          ⊢ LE.le (HSMul.hSMul c (m s)) (HSMul.hSMul c (m t))
        -/
        rw [← smul_one_mul c, ← smul_one_mul c (m t)]
        /-
          α : Type u_1
          β : Type u_2
          m✝ : MeasureTheory.OuterMeasure α
          R : Type u_3
          inst✝³ : SMul R ENNReal
          inst✝² : IsScalarTower R ENNReal ENNReal
          R' : Type u_4
          inst✝¹ : SMul R' ENNReal
          inst✝ : IsScalarTower R' ENNReal ENNReal
          c : R
          m : MeasureTheory.OuterMeasure α
          s t : Set α
          h : HasSubset.Subset s t
          ⊢ LE.le (HMul.hMul (HSMul.hSMul c 1) (m s)) (HMul.hMul (HSMul.hSMul c 1) (m t))
        -/
        exact mul_left_mono (m.mono h)
        /-
          🎉 no goals
        -/
      iUnion_nat := fun s _ => by
        /-
          α : Type u_1
          β : Type u_2
          m✝ : MeasureTheory.OuterMeasure α
          R : Type u_3
          inst✝³ : SMul R ENNReal
          inst✝² : IsScalarTower R ENNReal ENNReal
          R' : Type u_4
          inst✝¹ : SMul R' ENNReal
          inst✝ : IsScalarTower R' ENNReal ENNReal
          c : R
          m : MeasureTheory.OuterMeasure α
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ⊢ LE.le ((fun s => HSMul.hSMul c (m s)) (Set.iUnion fun i => s i)) (tsum fun i …
        -/
        simp_rw [← smul_one_mul c (m _), ENNReal.tsum_mul_left]
        /-
          α : Type u_1
          β : Type u_2
          m✝ : MeasureTheory.OuterMeasure α
          R : Type u_3
          inst✝³ : SMul R ENNReal
          inst✝² : IsScalarTower R ENNReal ENNReal
          R' : Type u_4
          inst✝¹ : SMul R' ENNReal
          inst✝ : IsScalarTower R' ENNReal ENNReal
          c : R
          m : MeasureTheory.OuterMeasure α
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ⊢ LE.le (HMul.hMul (HSMul.hSMul c 1) (m (Set.iUnion fun i => s i))) (HMul.hMul …
        -/
        exact mul_left_mono (measure_iUnion_le _) }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_smul (c : R) (m : OuterMeasure α) : ⇑(c • m) = c • ⇑m :=
  rfl


theorem smul_apply (c : R) (m : OuterMeasure α) (s : Set α) : (c • m) s = c • m s :=
  rfl


instance instSMulCommClass [SMulCommClass R R' ℝ≥0∞] : SMulCommClass R R' (OuterMeasure α) :=
  ⟨fun _ _ _ => ext fun _ => smul_comm _ _ _⟩


instance instIsScalarTower [SMul R R'] [IsScalarTower R R' ℝ≥0∞] :
    IsScalarTower R R' (OuterMeasure α) :=
  ⟨fun _ _ _ => ext fun _ => smul_assoc _ _ _⟩


instance instIsCentralScalar [SMul Rᵐᵒᵖ ℝ≥0∞] [IsCentralScalar R ℝ≥0∞] :
    IsCentralScalar R (OuterMeasure α) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


instance instMulAction {R : Type*} [Monoid R] [MulAction R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] :
    MulAction R (OuterMeasure α) :=
  Injective.mulAction _ coe_fn_injective coe_smul


instance addCommMonoid : AddCommMonoid (OuterMeasure α) :=
  Injective.addCommMonoid (show OuterMeasure α → Set α → ℝ≥0∞ from _) coe_fn_injective rfl
    (fun _ _ => rfl) fun _ _ => rfl


/-- `(⇑)` as an `AddMonoidHom`. -/
@[simps]
def coeFnAddMonoidHom : OuterMeasure α →+ Set α → ℝ≥0∞ where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


instance instDistribMulAction {R : Type*} [Monoid R] [DistribMulAction R ℝ≥0∞]
    [IsScalarTower R ℝ≥0∞ ℝ≥0∞] :
    DistribMulAction R (OuterMeasure α) :=
  Injective.distribMulAction coeFnAddMonoidHom coe_fn_injective coe_smul


instance instModule {R : Type*} [Semiring R] [Module R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] :
    Module R (OuterMeasure α) :=
  Injective.module R coeFnAddMonoidHom coe_fn_injective coe_smul


instance instBot : Bot (OuterMeasure α) :=
  ⟨0⟩


@[simp]
theorem coe_bot : (⊥ : OuterMeasure α) = 0 :=
  rfl


instance instPartialOrder : PartialOrder (OuterMeasure α) where
  le m₁ m₂ := ∀ s, m₁ s ≤ m₂ s
  le_refl _ _ := le_rfl
  le_trans _ _ _ hab hbc s := le_trans (hab s) (hbc s)
  le_antisymm _ _ hab hba := ext fun s => le_antisymm (hab s) (hba s)


instance orderBot : OrderBot (OuterMeasure α) :=
  { bot := 0,
                            /-
                              α : Type u_1
                              β : Type u_2
                              m a : MeasureTheory.OuterMeasure α
                              s : Set α
                              ⊢ LE.le (Bot.bot s) (a s)
                            -/
    bot_le := fun a s => by simp only [coe_zero, Pi.zero_apply, coe_bot, zero_le] }
                            /-
                              🎉 no goals
                            -/


theorem univ_eq_zero_iff (m : OuterMeasure α) : m univ = 0 ↔ m = 0 :=
  ⟨fun h => bot_unique fun s => (measure_mono <| subset_univ s).trans_eq h, fun h => h.symm ▸ rfl⟩


instance instSupSet : SupSet (OuterMeasure α) :=
  ⟨fun ms =>
    { measureOf := fun s => ⨆ m ∈ ms, (m : OuterMeasure α) s
      empty := nonpos_iff_eq_zero.1 <| iSup₂_le fun m _ => le_of_eq m.empty
      mono := fun {_ _} hs => iSup₂_mono fun m _ => m.mono hs
      iUnion_nat := fun f _ =>
        iSup₂_le fun m hm =>
          calc
            m (⋃ i, f i) ≤ ∑' i : ℕ, m (f i) := measure_iUnion_le _
            _ ≤ ∑' i, ⨆ m ∈ ms, (m : OuterMeasure α) (f i) :=
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  m✝ : MeasureTheory.OuterMeasure α
                                                  ms : Set (MeasureTheory.OuterMeasure α)
                                                  f : Nat → Set α
                                                  x✝ : Pairwise (Function.onFun Disjoint f)
                                                  m : MeasureTheory.OuterMeasure α
                                                  hm : Membership.mem ms m
                                                  i : Nat
                                                  ⊢ LE.le (m (f i)) (iSup fun m => iSup fun h => m (f i))
                                                -/
               ENNReal.tsum_le_tsum fun i => by apply le_iSup₂ m hm
                                                /-
                                                  🎉 no goals
                                                -/
             }⟩


instance instCompleteLattice : CompleteLattice (OuterMeasure α) :=
  { OuterMeasure.orderBot,
    completeLatticeOfSup (OuterMeasure α) fun ms =>
                        /-
                          α : Type u_1
                          β : Type u_2
                          m✝ : MeasureTheory.OuterMeasure α
                          ms : Set (MeasureTheory.OuterMeasure α)
                          m : MeasureTheory.OuterMeasure α
                          hm : Membership.mem ms m
                          s : Set α
                          ⊢ LE.le (m s) ((SupSet.sSup ms) s)
                        -/
      ⟨fun m hm s => by apply le_iSup₂ m hm, fun _ hm s => iSup₂_le fun _ hm' => hm hm' s⟩ with }
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem sSup_apply (ms : Set (OuterMeasure α)) (s : Set α) :
    (sSup ms) s = ⨆ m ∈ ms, (m : OuterMeasure α) s :=
  rfl


@[simp]
theorem iSup_apply {ι} (f : ι → OuterMeasure α) (s : Set α) : (⨆ i : ι, f i) s = ⨆ i, f i s := by
  /-
    α : Type u_1
    ι : Sort u_3
    f : ι → MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq ((iSup fun i => f i) s) (iSup fun i => (f i) s)
  -/
  rw [iSup, sSup_apply, iSup_range]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_iSup {ι} (f : ι → OuterMeasure α) : ⇑(⨆ i, f i) = ⨆ i, ⇑(f i) :=
                     /-
                       α : Type u_1
                       ι : Sort u_3
                       f : ι → MeasureTheory.OuterMeasure α
                       s : Set α
                       ⊢ Eq ((iSup fun i => f i) s) (iSup (fun i => ⇑(f i)) s)
                     -/
  funext fun s => by simp
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem sup_apply (m₁ m₂ : OuterMeasure α) (s : Set α) : (m₁ ⊔ m₂) s = m₁ s ⊔ m₂ s := by
  /-
    α : Type u_1
    m₁ m₂ : MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq ((Max.max m₁ m₂) s) (Max.max (m₁ s) (m₂ s))
  -/
  have := iSup_apply (fun b => cond b m₁ m₂) s; rwa [iSup_bool_eq, iSup_bool_eq] at this
                                                /-
                                                  🎉 no goals
                                                -/


theorem smul_iSup {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    {ι : Sort*} (f : ι → OuterMeasure α) (c : R) :
    (c • ⨆ i, f i) = ⨆ i, c • f i :=
                  /-
                    α : Type u_1
                    R : Type u_3
                    inst✝¹ : SMul R ENNReal
                    inst✝ : IsScalarTower R ENNReal ENNReal
                    ι : Sort u_4
                    f : ι → MeasureTheory.OuterMeasure α
                    c : R
                    s : Set α
                    ⊢ Eq ((HSMul.hSMul c (iSup fun i => f i)) s) ((iSup fun i => HSMul.hSMul c (f  …
                  -/
  ext fun s => by simp only [smul_apply, iSup_apply, ENNReal.smul_iSup]
                  /-
                    🎉 no goals
                  -/


@[mono, gcongr]
theorem mono'' {m₁ m₂ : OuterMeasure α} {s₁ s₂ : Set α} (hm : m₁ ≤ m₂) (hs : s₁ ⊆ s₂) :
    m₁ s₁ ≤ m₂ s₂ :=
  (hm s₁).trans (m₂.mono hs)


/-- The pushforward of `m` along `f`. The outer measure on `s` is defined to be `m (f ⁻¹' s)`. -/
def map {β} (f : α → β) : OuterMeasure α →ₗ[ℝ≥0∞] OuterMeasure β where
  toFun m :=
    { measureOf := fun s => m (f ⁻¹' s)
      empty := m.empty
      mono := fun {_ _} h => m.mono (preimage_mono h)
                                  /-
                                    α : Type u_1
                                    β✝ : Type u_2
                                    m✝ : MeasureTheory.OuterMeasure α
                                    β : Type ?u.32003
                                    f : α → β
                                    m : MeasureTheory.OuterMeasure α
                                    s : Nat → Set β
                                    x✝ : Pairwise (Function.onFun Disjoint s)
                                    ⊢ LE.le ((fun s => m (Set.preimage f s)) (Set.iUnion fun i => s i)) (tsum fun  …
                                  -/
      iUnion_nat := fun s _ => by simpa using measure_iUnion_le fun i => f ⁻¹' s i }
                                  /-
                                    🎉 no goals
                                  -/
  map_add' _ _ := coe_fn_injective rfl
  map_smul' _ _ := coe_fn_injective rfl


@[simp]
theorem map_apply {β} (f : α → β) (m : OuterMeasure α) (s : Set β) : map f m s = m (f ⁻¹' s) :=
  rfl


@[simp]
theorem map_id (m : OuterMeasure α) : map id m = m :=
  ext fun _ => rfl


@[simp]
theorem map_map {β γ} (f : α → β) (g : β → γ) (m : OuterMeasure α) :
    map g (map f m) = map (g ∘ f) m :=
  ext fun _ => rfl


@[mono]
theorem map_mono {β} (f : α → β) : Monotone (map f) := fun _ _ h _ => h _


@[simp]
theorem map_sup {β} (f : α → β) (m m' : OuterMeasure α) : map f (m ⊔ m') = map f m ⊔ map f m' :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    f : α → β
                    m m' : MeasureTheory.OuterMeasure α
                    s : Set β
                    ⊢ Eq (((MeasureTheory.OuterMeasure.map f) (Max.max m m')) s) ((Max.max ((Measu …
                  -/
  ext fun s => by simp only [map_apply, sup_apply]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem map_iSup {β ι} (f : α → β) (m : ι → OuterMeasure α) : map f (⨆ i, m i) = ⨆ i, map f (m i) :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    ι : Sort u_4
                    f : α → β
                    m : ι → MeasureTheory.OuterMeasure α
                    s : Set β
                    ⊢ Eq (((MeasureTheory.OuterMeasure.map f) (iSup fun i => m i)) s) ((iSup fun i …
                  -/
  ext fun s => by simp only [map_apply, iSup_apply]
                  /-
                    🎉 no goals
                  -/


instance instFunctor : Functor OuterMeasure where map {_ _} f := map f


                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                m : MeasureTheory.OuterMeasure α
                                                                ⊢ LawfulFunctor MeasureTheory.OuterMeasure
                                                              -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
instance instLawfulFunctor : LawfulFunctor OuterMeasure := by constructor <;> intros <;> rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


/-- The dirac outer measure. -/
def dirac (a : α) : OuterMeasure α where
  measureOf s := indicator s (fun _ => 1) a
              /-
                α : Type u_1
                β : Type u_2
                m : MeasureTheory.OuterMeasure α
                a : α
                ⊢ Eq ((fun s => s.indicator (fun x => 1) a) EmptyCollection.emptyCollection) 0
              -/
  empty := by simp
              /-
                🎉 no goals
              -/
  mono {_ _} h := indicator_le_indicator_of_subset h (fun _ => zero_le _) a
  iUnion_nat s _ := calc
    indicator (⋃ n, s n) 1 a = ⨆ n, indicator (s n) 1 a :=
      indicator_iUnion_apply (M := ℝ≥0∞) rfl _ _ _
    _ ≤ ∑' n, indicator (s n) 1 a := iSup_le fun _ ↦ ENNReal.le_tsum _


@[simp]
theorem dirac_apply (a : α) (s : Set α) : dirac a s = indicator s (fun _ => 1) a :=
  rfl


/-- The sum of an (arbitrary) collection of outer measures. -/
def sum {ι} (f : ι → OuterMeasure α) : OuterMeasure α where
  measureOf s := ∑' i, f i s
              /-
                α : Type u_1
                β : Type u_2
                m : MeasureTheory.OuterMeasure α
                ι : Type ?u.45226
                f : ι → MeasureTheory.OuterMeasure α
                ⊢ Eq ((fun s => tsum fun i => (f i) s) EmptyCollection.emptyCollection) 0
              -/
  empty := by simp
              /-
                🎉 no goals
              -/
  mono {_ _} h := ENNReal.tsum_le_tsum fun _ => measure_mono h
  iUnion_nat s _ := by
    /-
      α : Type u_1
      β : Type u_2
      m : MeasureTheory.OuterMeasure α
      ι : Type ?u.45226
      f : ι → MeasureTheory.OuterMeasure α
      s : Nat → Set α
      x✝ : Pairwise (Function.onFun Disjoint s)
      ⊢ LE.le ((fun s => tsum fun i => (f i) s) (Set.iUnion fun i => s i)) (tsum fun …
    -/
    rw [ENNReal.tsum_comm]; exact ENNReal.tsum_le_tsum fun i => measure_iUnion_le _
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem sum_apply {ι} (f : ι → OuterMeasure α) (s : Set α) : sum f s = ∑' i, f i s :=
  rfl


theorem smul_dirac_apply (a : ℝ≥0∞) (b : α) (s : Set α) :
    (a • dirac b) s = indicator s (fun _ => a) b := by
  /-
    α : Type u_1
    a : ENNReal
    b : α
    s : Set α
    ⊢ Eq ((HSMul.hSMul a (MeasureTheory.OuterMeasure.dirac b)) s) (s.indicator (fu …
  -/
  simp only [smul_apply, smul_eq_mul, dirac_apply, ← indicator_mul_right _ fun _ => a, mul_one]
  /-
    🎉 no goals
  -/


/-- Pullback of an `OuterMeasure`: `comap f μ s = μ (f '' s)`. -/
def comap {β} (f : α → β) : OuterMeasure β →ₗ[ℝ≥0∞] OuterMeasure α where
  toFun m :=
    { measureOf := fun s => m (f '' s)
                  /-
                    α : Type u_1
                    β✝ : Type u_2
                    m✝ : MeasureTheory.OuterMeasure α
                    β : Type ?u.49487
                    f : α → β
                    m : MeasureTheory.OuterMeasure β
                    ⊢ Eq ((fun s => m (Set.image f s)) EmptyCollection.emptyCollection) 0
                  -/
      empty := by simp
                  /-
                    🎉 no goals
                  -/
      mono := fun {_ _} h => m.mono <| image_subset f h
                                  /-
                                    α : Type u_1
                                    β✝ : Type u_2
                                    m✝ : MeasureTheory.OuterMeasure α
                                    β : Type ?u.49487
                                    f : α → β
                                    m : MeasureTheory.OuterMeasure β
                                    s : Nat → Set α
                                    x✝ : Pairwise (Function.onFun Disjoint s)
                                    ⊢ LE.le ((fun s => m (Set.image f s)) (Set.iUnion fun i => s i)) (tsum fun i = …
                                  -/
      iUnion_nat := fun s _ => by simpa only [image_iUnion] using measure_iUnion_le _ }
                                  /-
                                    🎉 no goals
                                  -/
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem comap_apply {β} (f : α → β) (m : OuterMeasure β) (s : Set α) : comap f m s = m (f '' s) :=
  rfl


@[mono]
theorem comap_mono {β} (f : α → β) : Monotone (comap f) := fun _ _ h _ => h _


@[simp]
theorem comap_iSup {β ι} (f : α → β) (m : ι → OuterMeasure β) :
    comap f (⨆ i, m i) = ⨆ i, comap f (m i) :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    ι : Sort u_4
                    f : α → β
                    m : ι → MeasureTheory.OuterMeasure β
                    s : Set α
                    ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) (iSup fun i => m i)) s) ((iSup fun …
                  -/
  ext fun s => by simp only [comap_apply, iSup_apply]
                  /-
                    🎉 no goals
                  -/


/-- Restrict an `OuterMeasure` to a set. -/
def restrict (s : Set α) : OuterMeasure α →ₗ[ℝ≥0∞] OuterMeasure α :=
  (map (↑)).comp (comap ((↑) : s → α))

-- TODO (kmill): change `m (t ∩ s)` to `m (s ∩ t)`

@[simp]
theorem restrict_apply (s t : Set α) (m : OuterMeasure α) : restrict s m t = m (t ∩ s) := by
  /-
    α : Type u_1
    s t : Set α
    m : MeasureTheory.OuterMeasure α
    ⊢ Eq (((MeasureTheory.OuterMeasure.restrict s) m) t) (m (Inter.inter t s))
  -/
  simp [restrict, inter_comm t]
  /-
    🎉 no goals
  -/


@[mono]
theorem restrict_mono {s t : Set α} (h : s ⊆ t) {m m' : OuterMeasure α} (hm : m ≤ m') :
    restrict s m ≤ restrict t m' := fun u => by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    m m' : MeasureTheory.OuterMeasure α
    hm : LE.le m m'
    u : Set α
    ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) m) u) (((MeasureTheory.Outer …
  -/
  simp only [restrict_apply]
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    m m' : MeasureTheory.OuterMeasure α
    hm : LE.le m m'
    u : Set α
    ⊢ LE.le (m (Inter.inter u s)) (m' (Inter.inter u t))
  -/
  exact (hm _).trans (m'.mono <| inter_subset_inter_right _ h)
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_univ (m : OuterMeasure α) : restrict univ m = m :=
                  /-
                    α : Type u_1
                    m : MeasureTheory.OuterMeasure α
                    s : Set α
                    ⊢ Eq (((MeasureTheory.OuterMeasure.restrict Set.univ) m) s) (m s)
                  -/
  ext fun s => by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem restrict_empty (m : OuterMeasure α) : restrict ∅ m = 0 :=
                  /-
                    α : Type u_1
                    m : MeasureTheory.OuterMeasure α
                    s : Set α
                    ⊢ Eq (((MeasureTheory.OuterMeasure.restrict EmptyCollection.emptyCollection) m …
                  -/
  ext fun s => by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem restrict_iSup {ι} (s : Set α) (m : ι → OuterMeasure α) :
                                                        /-
                                                          α : Type u_1
                                                          ι : Sort u_3
                                                          s : Set α
                                                          m : ι → MeasureTheory.OuterMeasure α
                                                          ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (iSup fun i => m i)) (iSup fun i …
                                                        -/
    restrict s (⨆ i, m i) = ⨆ i, restrict s (m i) := by simp [restrict]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem map_comap {β} (f : α → β) (m : OuterMeasure β) : map f (comap f m) = restrict (range f) m :=
                                 /-
                                   α : Type u_1
                                   β : Type u_3
                                   f : α → β
                                   m : MeasureTheory.OuterMeasure β
                                   s : Set β
                                   ⊢ Eq (Set.image f (Set.preimage f s)) (Set.image Subtype.val (Set.preimage Sub …
                                 -/
  ext fun s => congr_arg m <| by simp only [image_preimage_eq_inter_range, Subtype.range_coe]
                                 /-
                                   🎉 no goals
                                 -/


theorem map_comap_le {β} (f : α → β) (m : OuterMeasure β) : map f (comap f m) ≤ m := fun _ =>
  m.mono <| image_preimage_subset _ _


theorem restrict_le_self (m : OuterMeasure α) (s : Set α) : restrict s m ≤ m :=
  map_comap_le _ _


@[simp]
theorem map_le_restrict_range {β} {ma : OuterMeasure α} {mb : OuterMeasure β} {f : α → β} :
    map f ma ≤ restrict (range f) mb ↔ map f ma ≤ mb :=
                                                          /-
                                                            α : Type u_1
                                                            β : Type u_3
                                                            ma : MeasureTheory.OuterMeasure α
                                                            mb : MeasureTheory.OuterMeasure β
                                                            f : α → β
                                                            h : LE.le ((MeasureTheory.OuterMeasure.map f) ma) mb
                                                            s : Set β
                                                            ⊢ LE.le (((MeasureTheory.OuterMeasure.map f) ma) s) (((MeasureTheory.OuterMeas …
                                                          -/
  ⟨fun h => h.trans (restrict_le_self _ _), fun h s => by simpa using h (s ∩ range f)⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem map_comap_of_surjective {β} {f : α → β} (hf : Surjective f) (m : OuterMeasure β) :
    map f (comap f m) = m :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    f : α → β
                    hf : Function.Surjective f
                    m : MeasureTheory.OuterMeasure β
                    s : Set β
                    ⊢ Eq (((MeasureTheory.OuterMeasure.map f) ((MeasureTheory.OuterMeasure.comap f …
                  -/
  ext fun s => by rw [map_apply, comap_apply, hf.image_preimage]
                  /-
                    🎉 no goals
                  -/


theorem le_comap_map {β} (f : α → β) (m : OuterMeasure α) : m ≤ comap f (map f m) := fun _ =>
  m.mono <| subset_preimage_image _ _


theorem comap_map {β} {f : α → β} (hf : Injective f) (m : OuterMeasure α) : comap f (map f m) = m :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    f : α → β
                    hf : Function.Injective f
                    m : MeasureTheory.OuterMeasure α
                    s : Set α
                    ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) ((MeasureTheory.OuterMeasure.map f …
                  -/
  ext fun s => by rw [comap_apply, map_apply, hf.preimage_image]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem top_apply {s : Set α} (h : s.Nonempty) : (⊤ : OuterMeasure α) s = ∞ :=
  let ⟨a, as⟩ := h
                             /-
                               α : Type u_1
                               s : Set α
                               h : s.Nonempty
                               a : α
                               as : Membership.mem s a
                               ⊢ LE.le Top.top ((HSMul.hSMul Top.top (MeasureTheory.OuterMeasure.dirac a)) s)
                             -/
  top_unique <| le_trans (by simp [smul_dirac_apply, as]) (le_iSup₂ (∞ • dirac a) trivial)
                             /-
                               🎉 no goals
                             -/


theorem top_apply' (s : Set α) : (⊤ : OuterMeasure α) s = ⨅ _ : s = ∅, 0 :=
                                           /-
                                             α : Type u_1
                                             s : Set α
                                             h : Eq s EmptyCollection.emptyCollection
                                             ⊢ Eq (Top.top s) (iInf fun x => 0)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  s.eq_empty_or_nonempty.elim (fun h => by simp [h]) fun h => by simp [h, h.ne_empty]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem comap_top (f : α → β) : comap f ⊤ = ⊤ :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                s : Set α
                                hs : s.Nonempty
                                ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) Top.top) s) (Top.top s)
                              -/
  ext_nonempty fun s hs => by rw [comap_apply, top_apply hs, top_apply (hs.image _)]
                              /-
                                🎉 no goals
                              -/


theorem map_top (f : α → β) : map f ⊤ = restrict (range f) ⊤ :=
  ext fun s => by
    rw [map_apply, restrict_apply, ← image_preimage_eq_inter_range, top_apply', top_apply',
      Set.image_eq_empty]


@[simp]
theorem map_top_of_surjective (f : α → β) (hf : Surjective f) : map f ⊤ = ⊤ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) Top.top) Top.top
  -/
  rw [map_top, hf.range_eq, restrict_univ]
  /-
    🎉 no goals
  -/


