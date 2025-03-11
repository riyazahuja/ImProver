/-- We say that `α` is a `SupConvergenceClass` if the following holds. Let `f : ι → α` be a
monotone function, let `a : α` be a least upper bound of `Set.range f`. Then `f x` tends to `𝓝 a`
 as `x → ∞` (formally, at the filter `Filter.atTop`). We require this for `ι = (s : Set α)`,
`f = CoeTC.coe` in the definition, then prove it for any `f` in `tendsto_atTop_isLUB`.

This property holds for linear orders with order topology as well as their products. -/
class SupConvergenceClass (α : Type*) [Preorder α] [TopologicalSpace α] : Prop where
  /-- proof that a monotone function tends to `𝓝 a` as `x → ∞` -/
  tendsto_coe_atTop_isLUB :
    ∀ (a : α) (s : Set α), IsLUB s a → Tendsto (CoeTC.coe : s → α) atTop (𝓝 a)


/-- We say that `α` is an `InfConvergenceClass` if the following holds. Let `f : ι → α` be a
monotone function, let `a : α` be a greatest lower bound of `Set.range f`. Then `f x` tends to `𝓝 a`
as `x → -∞` (formally, at the filter `Filter.atBot`). We require this for `ι = (s : Set α)`,
`f = CoeTC.coe` in the definition, then prove it for any `f` in `tendsto_atBot_isGLB`.

This property holds for linear orders with order topology as well as their products. -/
class InfConvergenceClass (α : Type*) [Preorder α] [TopologicalSpace α] : Prop where
  /-- proof that a monotone function tends to `𝓝 a` as `x → -∞`-/
  tendsto_coe_atBot_isGLB :
    ∀ (a : α) (s : Set α), IsGLB s a → Tendsto (CoeTC.coe : s → α) atBot (𝓝 a)


instance OrderDual.supConvergenceClass [Preorder α] [TopologicalSpace α] [InfConvergenceClass α] :
    SupConvergenceClass αᵒᵈ :=
  ⟨‹InfConvergenceClass α›.1⟩


instance OrderDual.infConvergenceClass [Preorder α] [TopologicalSpace α] [SupConvergenceClass α] :
    InfConvergenceClass αᵒᵈ :=
  ⟨‹SupConvergenceClass α›.1⟩

-- see Note [lower instance priority]

instance (priority := 100) LinearOrder.supConvergenceClass [TopologicalSpace α] [LinearOrder α]
    [OrderTopology α] : SupConvergenceClass α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    ⊢ SupConvergenceClass α
  -/
  refine ⟨fun a s ha => tendsto_order.2 ⟨fun b hb => ?_, fun b hb => ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      b : α
      hb : LT.lt b a
      ⊢ Filter.Eventually (fun b_1 => LT.lt b (CoeTC.coe b_1)) Filter.atTop
    -/
  · rcases ha.exists_between hb with ⟨c, hcs, bc, bca⟩
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      b : α
      hb : LT.lt b a
      c : α
      hcs : Membership.mem s c
      bc : LT.lt b c
      bca : LE.le c a
      ⊢ Filter.Eventually (fun b_1 => LT.lt b (CoeTC.coe b_1)) Filter.atTop
    -/
    lift c to s using hcs
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      b : α
      hb : LT.lt b a
      c : Subtype fun x => Membership.mem s x
      bc : LT.lt b ↑c
      bca : LE.le (↑c) a
      ⊢ Filter.Eventually (fun b_1 => LT.lt b (CoeTC.coe b_1)) Filter.atTop
    -/
    exact (eventually_ge_atTop c).mono fun x hx => bc.trans_le hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      b : α
      hb : GT.gt b a
      ⊢ Filter.Eventually (fun b_1 => LT.lt (CoeTC.coe b_1) b) Filter.atTop
    -/
  · exact Eventually.of_forall fun x => (ha.1 x.2).trans_lt hb
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

instance (priority := 100) LinearOrder.infConvergenceClass [TopologicalSpace α] [LinearOrder α]
    [OrderTopology α] : InfConvergenceClass α :=
  show InfConvergenceClass αᵒᵈᵒᵈ from OrderDual.infConvergenceClass


theorem tendsto_atTop_isLUB (h_mono : Monotone f) (ha : IsLUB (Set.range f) a) :
    Tendsto f atTop (𝓝 a) := by
  suffices Tendsto (rangeFactorization f) atTop atTop from
    (SupConvergenceClass.tendsto_coe_atTop_isLUB _ _ ha).comp this
  /-
    α : Type u_1
    ι : Type u_3
    inst✝³ : Preorder ι
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : SupConvergenceClass α
    f : ι → α
    a : α
    h_mono : Monotone f
    ha : IsLUB (Set.range f) a
    ⊢ Filter.Tendsto (Set.rangeFactorization f) Filter.atTop Filter.atTop
  -/
  exact h_mono.rangeFactorization.tendsto_atTop_atTop fun b => b.2.imp fun a ha => ha.ge
  /-
    🎉 no goals
  -/


theorem tendsto_atBot_isLUB (h_anti : Antitone f) (ha : IsLUB (Set.range f) a) :
                                /-
                                  α : Type u_1
                                  ι : Type u_3
                                  inst✝³ : Preorder ι
                                  inst✝² : TopologicalSpace α
                                  inst✝¹ : Preorder α
                                  inst✝ : SupConvergenceClass α
                                  f : ι → α
                                  a : α
                                  h_anti : Antitone f
                                  ha : IsLUB (Set.range f) a
                                  ⊢ Filter.Tendsto f Filter.atBot (nhds a)
                                -/
    Tendsto f atBot (𝓝 a) := by convert tendsto_atTop_isLUB h_anti.dual_left ha using 1
                                /-
                                  🎉 no goals
                                -/


theorem tendsto_atBot_isGLB (h_mono : Monotone f) (ha : IsGLB (Set.range f) a) :
                                /-
                                  α : Type u_1
                                  ι : Type u_3
                                  inst✝³ : Preorder ι
                                  inst✝² : TopologicalSpace α
                                  inst✝¹ : Preorder α
                                  inst✝ : InfConvergenceClass α
                                  f : ι → α
                                  a : α
                                  h_mono : Monotone f
                                  ha : IsGLB (Set.range f) a
                                  ⊢ Filter.Tendsto f Filter.atBot (nhds a)
                                -/
    Tendsto f atBot (𝓝 a) := by convert tendsto_atTop_isLUB h_mono.dual ha.dual using 1
                                /-
                                  🎉 no goals
                                -/


theorem tendsto_atTop_isGLB (h_anti : Antitone f) (ha : IsGLB (Set.range f) a) :
                                /-
                                  α : Type u_1
                                  ι : Type u_3
                                  inst✝³ : Preorder ι
                                  inst✝² : TopologicalSpace α
                                  inst✝¹ : Preorder α
                                  inst✝ : InfConvergenceClass α
                                  f : ι → α
                                  a : α
                                  h_anti : Antitone f
                                  ha : IsGLB (Set.range f) a
                                  ⊢ Filter.Tendsto f Filter.atTop (nhds a)
                                -/
    Tendsto f atTop (𝓝 a) := by convert tendsto_atBot_isLUB h_anti.dual ha.dual using 1
                                /-
                                  🎉 no goals
                                -/


theorem tendsto_atTop_ciSup (h_mono : Monotone f) (hbdd : BddAbove <| range f) :
    Tendsto f atTop (𝓝 (⨆ i, f i)) := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝³ : Preorder ι
    inst✝² : TopologicalSpace α
    inst✝¹ : ConditionallyCompleteLattice α
    inst✝ : SupConvergenceClass α
    f : ι → α
    h_mono : Monotone f
    hbdd : BddAbove (Set.range f)
    ⊢ Filter.Tendsto f Filter.atTop (nhds (iSup fun i => f i))
  -/
  cases isEmpty_or_nonempty ι
  /-
    case inl
    α : Type u_1
    ι : Type u_3
    inst✝³ : Preorder ι
    inst✝² : TopologicalSpace α
    inst✝¹ : ConditionallyCompleteLattice α
    inst✝ : SupConvergenceClass α
    f : ι → α
    h_mono : Monotone f
    hbdd : BddAbove (Set.range f)
    h✝ : IsEmpty ι
    ⊢ Filter.Tendsto f Filter.atTop (nhds (iSup fun i => f i))
  -/
  exacts [tendsto_of_isEmpty, tendsto_atTop_isLUB h_mono (isLUB_ciSup hbdd)]
  /-
    🎉 no goals
  -/


theorem tendsto_atBot_ciSup (h_anti : Antitone f) (hbdd : BddAbove <| range f) :
                                         /-
                                           α : Type u_1
                                           ι : Type u_3
                                           inst✝³ : Preorder ι
                                           inst✝² : TopologicalSpace α
                                           inst✝¹ : ConditionallyCompleteLattice α
                                           inst✝ : SupConvergenceClass α
                                           f : ι → α
                                           h_anti : Antitone f
                                           hbdd : BddAbove (Set.range f)
                                           ⊢ Filter.Tendsto f Filter.atBot (nhds (iSup fun i => f i))
                                         -/
    Tendsto f atBot (𝓝 (⨆ i, f i)) := by convert tendsto_atTop_ciSup h_anti.dual hbdd.dual using 1
                                         /-
                                           🎉 no goals
                                         -/


theorem tendsto_atBot_ciInf (h_mono : Monotone f) (hbdd : BddBelow <| range f) :
                                         /-
                                           α : Type u_1
                                           ι : Type u_3
                                           inst✝³ : Preorder ι
                                           inst✝² : TopologicalSpace α
                                           inst✝¹ : ConditionallyCompleteLattice α
                                           inst✝ : InfConvergenceClass α
                                           f : ι → α
                                           h_mono : Monotone f
                                           hbdd : BddBelow (Set.range f)
                                           ⊢ Filter.Tendsto f Filter.atBot (nhds (iInf fun i => f i))
                                         -/
    Tendsto f atBot (𝓝 (⨅ i, f i)) := by convert tendsto_atTop_ciSup h_mono.dual hbdd.dual using 1
                                         /-
                                           🎉 no goals
                                         -/


theorem tendsto_atTop_ciInf (h_anti : Antitone f) (hbdd : BddBelow <| range f) :
                                         /-
                                           α : Type u_1
                                           ι : Type u_3
                                           inst✝³ : Preorder ι
                                           inst✝² : TopologicalSpace α
                                           inst✝¹ : ConditionallyCompleteLattice α
                                           inst✝ : InfConvergenceClass α
                                           f : ι → α
                                           h_anti : Antitone f
                                           hbdd : BddBelow (Set.range f)
                                           ⊢ Filter.Tendsto f Filter.atTop (nhds (iInf fun i => f i))
                                         -/
    Tendsto f atTop (𝓝 (⨅ i, f i)) := by convert tendsto_atBot_ciSup h_anti.dual hbdd.dual using 1
                                         /-
                                           🎉 no goals
                                         -/


theorem tendsto_atTop_iSup (h_mono : Monotone f) : Tendsto f atTop (𝓝 (⨆ i, f i)) :=
  tendsto_atTop_ciSup h_mono (OrderTop.bddAbove _)


theorem tendsto_atBot_iSup (h_anti : Antitone f) : Tendsto f atBot (𝓝 (⨆ i, f i)) :=
  tendsto_atBot_ciSup h_anti (OrderTop.bddAbove _)


theorem tendsto_atBot_iInf (h_mono : Monotone f) : Tendsto f atBot (𝓝 (⨅ i, f i)) :=
  tendsto_atBot_ciInf h_mono (OrderBot.bddBelow _)


theorem tendsto_atTop_iInf (h_anti : Antitone f) : Tendsto f atTop (𝓝 (⨅ i, f i)) :=
  tendsto_atTop_ciInf h_anti (OrderBot.bddBelow _)


instance Prod.supConvergenceClass
    [Preorder α] [Preorder β] [TopologicalSpace α] [TopologicalSpace β]
    [SupConvergenceClass α] [SupConvergenceClass β] : SupConvergenceClass (α × β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : SupConvergenceClass α
    inst✝ : SupConvergenceClass β
    ⊢ SupConvergenceClass (Prod α β)
  -/
  constructor
  /-
    case tendsto_coe_atTop_isLUB
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : SupConvergenceClass α
    inst✝ : SupConvergenceClass β
    ⊢ ∀ (a : Prod α β) (s : Set (Prod α β)), IsLUB s a → Filter.Tendsto CoeTC.coe  …
  -/
  rintro ⟨a, b⟩ s h
  /-
    case tendsto_coe_atTop_isLUB.mk
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : SupConvergenceClass α
    inst✝ : SupConvergenceClass β
    a : α
    b : β
    s : Set (Prod α β)
    h : IsLUB s { fst := a, snd := b }
    ⊢ Filter.Tendsto CoeTC.coe Filter.atTop (nhds { fst := a, snd := b })
  -/
  rw [isLUB_prod, ← range_restrict, ← range_restrict] at h
  have A : Tendsto (fun x : s => (x : α × β).1) atTop (𝓝 a) :=
    tendsto_atTop_isLUB (monotone_fst.restrict s) h.1
  have B : Tendsto (fun x : s => (x : α × β).2) atTop (𝓝 b) :=
    tendsto_atTop_isLUB (monotone_snd.restrict s) h.2
  /-
    case tendsto_coe_atTop_isLUB.mk
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : SupConvergenceClass α
    inst✝ : SupConvergenceClass β
    a : α
    b : β
    s : Set (Prod α β)
    h : And (IsLUB (Set.range (s.restrict Prod.fst)) { fst := a, snd := b }.1) (Is …
    A : Filter.Tendsto (fun x => (↑x).1) Filter.atTop (nhds a)
    B : Filter.Tendsto (fun x => (↑x).2) Filter.atTop (nhds b)
    ⊢ Filter.Tendsto CoeTC.coe Filter.atTop (nhds { fst := a, snd := b })
  -/
  convert A.prod_mk_nhds B
  /-
    🎉 no goals
  -/
  -- Porting note: previously required below to close
  -- ext1 ⟨⟨x, y⟩, h⟩
  -- rfl


instance [Preorder α] [Preorder β] [TopologicalSpace α] [TopologicalSpace β] [InfConvergenceClass α]
    [InfConvergenceClass β] : InfConvergenceClass (α × β) :=
  show InfConvergenceClass (αᵒᵈ × βᵒᵈ)ᵒᵈ from OrderDual.infConvergenceClass


instance Pi.supConvergenceClass
    {ι : Type*} {α : ι → Type*} [∀ i, Preorder (α i)] [∀ i, TopologicalSpace (α i)]
    [∀ i, SupConvergenceClass (α i)] : SupConvergenceClass (∀ i, α i) := by
  /-
    α✝ : Type u_1
    β : Type u_2
    ι : Type u_3
    α : ι → Type u_4
    inst✝² : (i : ι) → Preorder (α i)
    inst✝¹ : (i : ι) → TopologicalSpace (α i)
    inst✝ : ∀ (i : ι), SupConvergenceClass (α i)
    ⊢ SupConvergenceClass ((i : ι) → α i)
  -/
  refine ⟨fun f s h => ?_⟩
  /-
    α✝ : Type u_1
    β : Type u_2
    ι : Type u_3
    α : ι → Type u_4
    inst✝² : (i : ι) → Preorder (α i)
    inst✝¹ : (i : ι) → TopologicalSpace (α i)
    inst✝ : ∀ (i : ι), SupConvergenceClass (α i)
    f : (i : ι) → α i
    s : Set ((i : ι) → α i)
    h : IsLUB s f
    ⊢ Filter.Tendsto CoeTC.coe Filter.atTop (nhds f)
  -/
  simp only [isLUB_pi, ← range_restrict] at h
  /-
    α✝ : Type u_1
    β : Type u_2
    ι : Type u_3
    α : ι → Type u_4
    inst✝² : (i : ι) → Preorder (α i)
    inst✝¹ : (i : ι) → TopologicalSpace (α i)
    inst✝ : ∀ (i : ι), SupConvergenceClass (α i)
    f : (i : ι) → α i
    s : Set ((i : ι) → α i)
    h : ∀ (a : ι), IsLUB (Set.range (s.restrict (Function.eval a))) (f a)
    ⊢ Filter.Tendsto CoeTC.coe Filter.atTop (nhds f)
  -/
  exact tendsto_pi_nhds.2 fun i => tendsto_atTop_isLUB ((monotone_eval _).restrict _) (h i)
  /-
    🎉 no goals
  -/


instance Pi.infConvergenceClass
    {ι : Type*} {α : ι → Type*} [∀ i, Preorder (α i)] [∀ i, TopologicalSpace (α i)]
    [∀ i, InfConvergenceClass (α i)] : InfConvergenceClass (∀ i, α i) :=
  show InfConvergenceClass (∀ i, (α i)ᵒᵈ)ᵒᵈ from OrderDual.infConvergenceClass


instance Pi.supConvergenceClass' {ι : Type*} [Preorder α] [TopologicalSpace α]
    [SupConvergenceClass α] : SupConvergenceClass (ι → α) :=
  supConvergenceClass


instance Pi.infConvergenceClass' {ι : Type*} [Preorder α] [TopologicalSpace α]
    [InfConvergenceClass α] : InfConvergenceClass (ι → α) :=
  Pi.infConvergenceClass


theorem tendsto_of_monotone {ι α : Type*} [Preorder ι] [TopologicalSpace α]
    [ConditionallyCompleteLinearOrder α] [OrderTopology α] {f : ι → α} (h_mono : Monotone f) :
    Tendsto f atTop atTop ∨ ∃ l, Tendsto f atTop (𝓝 l) := by
  classical
  exact if H : BddAbove (range f) then Or.inr ⟨_, tendsto_atTop_ciSup h_mono H⟩
  else Or.inl <| tendsto_atTop_atTop_of_monotone' h_mono H


theorem tendsto_of_antitone {ι α : Type*} [Preorder ι] [TopologicalSpace α]
    [ConditionallyCompleteLinearOrder α] [OrderTopology α] {f : ι → α} (h_mono : Antitone f) :
    Tendsto f atTop atBot ∨ ∃ l, Tendsto f atTop (𝓝 l) :=
  @tendsto_of_monotone ι αᵒᵈ _ _ _ _ _ h_mono


theorem tendsto_iff_tendsto_subseq_of_monotone {ι₁ ι₂ α : Type*} [SemilatticeSup ι₁] [Preorder ι₂]
    [Nonempty ι₁] [TopologicalSpace α] [ConditionallyCompleteLinearOrder α] [OrderTopology α]
    [NoMaxOrder α] {f : ι₂ → α} {φ : ι₁ → ι₂} {l : α} (hf : Monotone f)
    (hg : Tendsto φ atTop atTop) : Tendsto f atTop (𝓝 l) ↔ Tendsto (f ∘ φ) atTop (𝓝 l) := by
  /-
    ι₁ : Type u_3
    ι₂ : Type u_4
    α : Type u_5
    inst✝⁶ : SemilatticeSup ι₁
    inst✝⁵ : Preorder ι₂
    inst✝⁴ : Nonempty ι₁
    inst✝³ : TopologicalSpace α
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : NoMaxOrder α
    f : ι₂ → α
    φ : ι₁ → ι₂
    l : α
    hf : Monotone f
    hg : Filter.Tendsto φ Filter.atTop Filter.atTop
    ⊢ Iff (Filter.Tendsto f Filter.atTop (nhds l)) (Filter.Tendsto (Function.comp  …
  -/
  constructor <;> intro h
    /-
      case mp
      ι₁ : Type u_3
      ι₂ : Type u_4
      α : Type u_5
      inst✝⁶ : SemilatticeSup ι₁
      inst✝⁵ : Preorder ι₂
      inst✝⁴ : Nonempty ι₁
      inst✝³ : TopologicalSpace α
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : NoMaxOrder α
      f : ι₂ → α
      φ : ι₁ → ι₂
      l : α
      hf : Monotone f
      hg : Filter.Tendsto φ Filter.atTop Filter.atTop
      h : Filter.Tendsto f Filter.atTop (nhds l)
      ⊢ Filter.Tendsto (Function.comp f φ) Filter.atTop (nhds l)
    -/
  · exact h.comp hg
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι₁ : Type u_3
      ι₂ : Type u_4
      α : Type u_5
      inst✝⁶ : SemilatticeSup ι₁
      inst✝⁵ : Preorder ι₂
      inst✝⁴ : Nonempty ι₁
      inst✝³ : TopologicalSpace α
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : NoMaxOrder α
      f : ι₂ → α
      φ : ι₁ → ι₂
      l : α
      hf : Monotone f
      hg : Filter.Tendsto φ Filter.atTop Filter.atTop
      h : Filter.Tendsto (Function.comp f φ) Filter.atTop (nhds l)
      ⊢ Filter.Tendsto f Filter.atTop (nhds l)
    -/
  · rcases tendsto_of_monotone hf with (h' | ⟨l', hl'⟩)
      /-
        case mpr.inl
        ι₁ : Type u_3
        ι₂ : Type u_4
        α : Type u_5
        inst✝⁶ : SemilatticeSup ι₁
        inst✝⁵ : Preorder ι₂
        inst✝⁴ : Nonempty ι₁
        inst✝³ : TopologicalSpace α
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : NoMaxOrder α
        f : ι₂ → α
        φ : ι₁ → ι₂
        l : α
        hf : Monotone f
        hg : Filter.Tendsto φ Filter.atTop Filter.atTop
        h : Filter.Tendsto (Function.comp f φ) Filter.atTop (nhds l)
        h' : Filter.Tendsto f Filter.atTop Filter.atTop
        ⊢ Filter.Tendsto f Filter.atTop (nhds l)
      -/
    · exact (not_tendsto_atTop_of_tendsto_nhds h (h'.comp hg)).elim
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        ι₁ : Type u_3
        ι₂ : Type u_4
        α : Type u_5
        inst✝⁶ : SemilatticeSup ι₁
        inst✝⁵ : Preorder ι₂
        inst✝⁴ : Nonempty ι₁
        inst✝³ : TopologicalSpace α
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : NoMaxOrder α
        f : ι₂ → α
        φ : ι₁ → ι₂
        l : α
        hf : Monotone f
        hg : Filter.Tendsto φ Filter.atTop Filter.atTop
        h : Filter.Tendsto (Function.comp f φ) Filter.atTop (nhds l)
        l' : α
        hl' : Filter.Tendsto f Filter.atTop (nhds l')
        ⊢ Filter.Tendsto f Filter.atTop (nhds l)
      -/
    · rwa [tendsto_nhds_unique h (hl'.comp hg)]
      /-
        🎉 no goals
      -/


theorem tendsto_iff_tendsto_subseq_of_antitone {ι₁ ι₂ α : Type*} [SemilatticeSup ι₁] [Preorder ι₂]
    [Nonempty ι₁] [TopologicalSpace α] [ConditionallyCompleteLinearOrder α] [OrderTopology α]
    [NoMinOrder α] {f : ι₂ → α} {φ : ι₁ → ι₂} {l : α} (hf : Antitone f)
    (hg : Tendsto φ atTop atTop) : Tendsto f atTop (𝓝 l) ↔ Tendsto (f ∘ φ) atTop (𝓝 l) :=
  tendsto_iff_tendsto_subseq_of_monotone (α := αᵒᵈ) hf hg


theorem Monotone.ge_of_tendsto [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [SemilatticeSup β] {f : β → α} {a : α} (hf : Monotone f) (ha : Tendsto f atTop (𝓝 a)) (b : β) :
    f b ≤ a :=
  haveI : Nonempty β := Nonempty.intro b
  _root_.ge_of_tendsto ha ((eventually_ge_atTop b).mono fun _ hxy => hf hxy)


theorem Monotone.le_of_tendsto [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [SemilatticeInf β] {f : β → α} {a : α} (hf : Monotone f) (ha : Tendsto f atBot (𝓝 a)) (b : β) :
    a ≤ f b :=
  hf.dual.ge_of_tendsto ha b


theorem Antitone.le_of_tendsto [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [SemilatticeSup β] {f : β → α} {a : α} (hf : Antitone f) (ha : Tendsto f atTop (𝓝 a)) (b : β) :
    a ≤ f b :=
  hf.dual_right.ge_of_tendsto ha b


theorem Antitone.ge_of_tendsto [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [SemilatticeInf β] {f : β → α} {a : α} (hf : Antitone f) (ha : Tendsto f atBot (𝓝 a)) (b : β) :
    f b ≤ a :=
  hf.dual_right.le_of_tendsto ha b


theorem isLUB_of_tendsto_atTop [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [Nonempty β] [SemilatticeSup β] {f : β → α} {a : α} (hf : Monotone f)
    (ha : Tendsto f atTop (𝓝 a)) : IsLUB (Set.range f) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : Preorder α
    inst✝² : OrderClosedTopology α
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    f : β → α
    a : α
    hf : Monotone f
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    ⊢ IsLUB (Set.range f) a
  -/
  constructor
    /-
      case left
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder α
      inst✝² : OrderClosedTopology α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      f : β → α
      a : α
      hf : Monotone f
      ha : Filter.Tendsto f Filter.atTop (nhds a)
      ⊢ Membership.mem (upperBounds (Set.range f)) a
    -/
  · rintro _ ⟨b, rfl⟩
    /-
      case left.intro
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder α
      inst✝² : OrderClosedTopology α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      f : β → α
      a : α
      hf : Monotone f
      ha : Filter.Tendsto f Filter.atTop (nhds a)
      b : β
      ⊢ LE.le (f b) a
    -/
    exact hf.ge_of_tendsto ha b
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : Preorder α
      inst✝² : OrderClosedTopology α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      f : β → α
      a : α
      hf : Monotone f
      ha : Filter.Tendsto f Filter.atTop (nhds a)
      ⊢ Membership.mem (lowerBounds (upperBounds (Set.range f))) a
    -/
  · exact fun _ hb => le_of_tendsto' ha fun x => hb (Set.mem_range_self x)
    /-
      🎉 no goals
    -/


theorem isGLB_of_tendsto_atBot [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [Nonempty β] [SemilatticeInf β] {f : β → α} {a : α} (hf : Monotone f)
    (ha : Tendsto f atBot (𝓝 a)) : IsGLB (Set.range f) a :=
  @isLUB_of_tendsto_atTop αᵒᵈ βᵒᵈ _ _ _ _ _ _ _ hf.dual ha


theorem isLUB_of_tendsto_atBot [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [Nonempty β] [SemilatticeInf β] {f : β → α} {a : α} (hf : Antitone f)
    (ha : Tendsto f atBot (𝓝 a)) : IsLUB (Set.range f) a :=
  @isLUB_of_tendsto_atTop α βᵒᵈ _ _ _ _ _ _ _ hf.dual_left ha


theorem isGLB_of_tendsto_atTop [TopologicalSpace α] [Preorder α] [OrderClosedTopology α]
    [Nonempty β] [SemilatticeSup β] {f : β → α} {a : α} (hf : Antitone f)
    (ha : Tendsto f atTop (𝓝 a)) : IsGLB (Set.range f) a :=
  @isGLB_of_tendsto_atBot α βᵒᵈ _ _ _ _ _ _ _ hf.dual_left ha


theorem iSup_eq_of_tendsto {α β} [TopologicalSpace α] [CompleteLinearOrder α] [OrderTopology α]
    [Nonempty β] [SemilatticeSup β] {f : β → α} {a : α} (hf : Monotone f) :
    Tendsto f atTop (𝓝 a) → iSup f = a :=
  tendsto_nhds_unique (tendsto_atTop_iSup hf)


theorem iInf_eq_of_tendsto {α} [TopologicalSpace α] [CompleteLinearOrder α] [OrderTopology α]
    [Nonempty β] [SemilatticeSup β] {f : β → α} {a : α} (hf : Antitone f) :
    Tendsto f atTop (𝓝 a) → iInf f = a :=
  tendsto_nhds_unique (tendsto_atTop_iInf hf)


theorem iSup_eq_iSup_subseq_of_monotone {ι₁ ι₂ α : Type*} [Preorder ι₂] [CompleteLattice α]
    {l : Filter ι₁} [l.NeBot] {f : ι₂ → α} {φ : ι₁ → ι₂} (hf : Monotone f)
    (hφ : Tendsto φ l atTop) : ⨆ i, f i = ⨆ i, f (φ i) :=
  le_antisymm
    (iSup_mono' fun i =>
      Exists.imp (fun j (hj : i ≤ φ j) => hf hj) (hφ.eventually <| eventually_ge_atTop i).exists)
    (iSup_mono' fun i => ⟨φ i, le_rfl⟩)


theorem iSup_eq_iSup_subseq_of_antitone {ι₁ ι₂ α : Type*} [Preorder ι₂] [CompleteLattice α]
    {l : Filter ι₁} [l.NeBot] {f : ι₂ → α} {φ : ι₁ → ι₂} (hf : Antitone f)
    (hφ : Tendsto φ l atBot) : ⨆ i, f i = ⨆ i, f (φ i) :=
  le_antisymm
    (iSup_mono' fun i =>
      Exists.imp (fun j (hj : φ j ≤ i) => hf hj) (hφ.eventually <| eventually_le_atBot i).exists)
    (iSup_mono' fun i => ⟨φ i, le_rfl⟩)


theorem iInf_eq_iInf_subseq_of_monotone {ι₁ ι₂ α : Type*} [Preorder ι₂] [CompleteLattice α]
    {l : Filter ι₁} [l.NeBot] {f : ι₂ → α} {φ : ι₁ → ι₂} (hf : Monotone f)
    (hφ : Tendsto φ l atBot) : ⨅ i, f i = ⨅ i, f (φ i) :=
  iSup_eq_iSup_subseq_of_monotone hf.dual hφ


theorem iInf_eq_iInf_subseq_of_antitone {ι₁ ι₂ α : Type*} [Preorder ι₂] [CompleteLattice α]
    {l : Filter ι₁} [l.NeBot] {f : ι₂ → α} {φ : ι₁ → ι₂} (hf : Antitone f)
    (hφ : Tendsto φ l atTop) : ⨅ i, f i = ⨅ i, f (φ i) :=
  iSup_eq_iSup_subseq_of_antitone hf.dual hφ

