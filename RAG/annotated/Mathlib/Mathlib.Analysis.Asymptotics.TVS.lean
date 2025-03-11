/-- `f =o[𝕜;l] g` (`IsLittleOTVS 𝕜 l f g`) is a generalization of `f =o[l] g` (`IsLittleO l f g`)
that works in topological `𝕜`-vector spaces.

Given two functions `f` and `g` taking values in topological vector spaces
over a normed field `K`,
we say that $f = o(g)$ if for any neighborhood of zero `U` in the codomain of `f`
there exists a neighborhood of zero `V` in the codomain of `g`
such that $\operatorname{gauge}_{K, U} (f(x)) = o(\operatorname{gauge}_{K, V} (g(x)))$,
where $\operatorname{gauge}_{K, U}(y) = \inf \{‖c‖ \mid y ∈ c • U\}$.

We use an `ENNReal`-valued function `egauge` for the gauge,
so we unfold the definition of little o instead of reusing it. -/
def IsLittleOTVS (𝕜 : Type*) {α E F : Type*}
    [NNNorm 𝕜] [TopologicalSpace E] [TopologicalSpace F] [Zero E] [Zero F] [SMul 𝕜 E] [SMul 𝕜 F]
    (l : Filter α) (f : α → E) (g : α → F) : Prop :=
  ∀ U ∈ 𝓝 (0 : E), ∃ V ∈ 𝓝 (0 : F), ∀ ε ≠ (0 : ℝ≥0),
    ∀ᶠ x in l, egauge 𝕜 U (f x) ≤ ε * egauge 𝕜 V (g x)


@[inherit_doc]
notation:100 f " =o[" 𝕜 ";" l "] " g:100 => IsLittleOTVS 𝕜 l f g


theorem _root_.Filter.HasBasis.isLittleOTVS_iff {ιE ιF : Sort*} {pE : ιE → Prop} {pF : ιF → Prop}
    {sE : ιE → Set E} {sF : ιF → Set F} (hE : HasBasis (𝓝 (0 : E)) pE sE)
    (hF : HasBasis (𝓝 (0 : F)) pF sF) {f : α → E} {g : α → F} {l : Filter α} :
    f =o[𝕜;l] g ↔ ∀ i, pE i → ∃ j, pF j ∧ ∀ ε ≠ (0 : ℝ≥0),
      ∀ᶠ x in l, egauge 𝕜 (sE i) (f x) ≤ ε * egauge 𝕜 (sF j) (g x) := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommGroup F
    inst✝¹ : TopologicalSpace F
    inst✝ : Module 𝕜 F
    ιE : Sort u_6
    ιF : Sort u_7
    pE : ιE → Prop
    pF : ιF → Prop
    sE : ιE → Set E
    sF : ιF → Set F
    hE : (nhds 0).HasBasis pE sE
    hF : (nhds 0).HasBasis pF sF
    f : α → E
    g : α → F
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 l f g) (∀ (i : ιE), pE i → Exists fun j => A …
  -/
  refine (hE.forall_iff ?_).trans <| forall₂_congr fun _ _ ↦ hF.exists_iff ?_
    /-
      case refine_1
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalSpace F
      inst✝ : Module 𝕜 F
      ιE : Sort u_6
      ιF : Sort u_7
      pE : ιE → Prop
      pF : ιF → Prop
      sE : ιE → Set E
      sF : ιF → Set F
      hE : (nhds 0).HasBasis pE sE
      hF : (nhds 0).HasBasis pF sF
      f : α → E
      g : α → F
      l : Filter α
      ⊢ ∀ ⦃s t : Set E⦄, HasSubset.Subset s t → (Exists fun V => And (Membership.mem …
    -/
  · rintro s t hsub ⟨V, hV₀, hV⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalSpace F
      inst✝ : Module 𝕜 F
      ιE : Sort u_6
      ιF : Sort u_7
      pE : ιE → Prop
      pF : ιF → Prop
      sE : ιE → Set E
      sF : ιF → Set F
      hE : (nhds 0).HasBasis pE sE
      hF : (nhds 0).HasBasis pF sF
      f : α → E
      g : α → F
      l : Filter α
      s t : Set E
      hsub : HasSubset.Subset s t
      V : Set F
      hV₀ : Membership.mem (nhds 0) V
      hV : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 s (f …
      ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (∀ (ε : NNReal), Ne ε 0 → Fi …
    -/
    exact ⟨V, hV₀, fun ε hε ↦ (hV ε hε).mono fun x ↦ le_trans <| egauge_anti _ hsub _⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalSpace F
      inst✝ : Module 𝕜 F
      ιE : Sort u_6
      ιF : Sort u_7
      pE : ιE → Prop
      pF : ιF → Prop
      sE : ιE → Set E
      sF : ιF → Set F
      hE : (nhds 0).HasBasis pE sE
      hF : (nhds 0).HasBasis pF sF
      f : α → E
      g : α → F
      l : Filter α
      x✝¹ : ιE
      x✝ : pE x✝¹
      ⊢ ∀ ⦃s t : Set F⦄, HasSubset.Subset s t → (∀ (ε : NNReal), Ne ε 0 → Filter.Eve …
    -/
  · refine fun s t hsub h ε hε ↦ (h ε hε).mono fun x hx ↦ hx.trans ?_
    /-
      case refine_2
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalSpace F
      inst✝ : Module 𝕜 F
      ιE : Sort u_6
      ιF : Sort u_7
      pE : ιE → Prop
      pF : ιF → Prop
      sE : ιE → Set E
      sF : ιF → Set F
      hE : (nhds 0).HasBasis pE sE
      hF : (nhds 0).HasBasis pF sF
      f : α → E
      g : α → F
      l : Filter α
      x✝¹ : ιE
      x✝ : pE x✝¹
      s t : Set F
      hsub : HasSubset.Subset s t
      h : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 (sE x …
      ε : NNReal
      hε : Ne ε 0
      x : α
      hx : LE.le (egauge 𝕜 (sE x✝¹) (f x)) (HMul.hMul (↑ε) (egauge 𝕜 t (g x)))
      ⊢ LE.le (HMul.hMul (↑ε) (egauge 𝕜 t (g x))) (HMul.hMul (↑ε) (egauge 𝕜 s (g x)))
    -/
    gcongr
    /-
      🎉 no goals
    -/


@[simp]
theorem isLittleOTVS_map {f : α → E} {g : α → F} {k : β → α} {l : Filter β} :
    f =o[𝕜; map k l] g ↔ (f ∘ k) =o[𝕜;l] (g ∘ k) := by
  /-
    α : Type u_1
    β : Type u_2
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommGroup F
    inst✝¹ : TopologicalSpace F
    inst✝ : Module 𝕜 F
    f : α → E
    g : α → F
    k : β → α
    l : Filter β
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 (Filter.map k l) f g) (Asymptotics.IsLittleO …
  -/
  simp [IsLittleOTVS]
  /-
    🎉 no goals
  -/


protected lemma IsLittleOTVS.smul_left {f : α → E} {g : α → F} {l : Filter α}
    (h : f =o[𝕜;l] g) (c : α → 𝕜) :
    (fun x ↦ c x • f x) =o[𝕜;l] (fun x ↦ c x • g x) := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommGroup F
    inst✝¹ : TopologicalSpace F
    inst✝ : Module 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    h : Asymptotics.IsLittleOTVS 𝕜 l f g
    c : α → 𝕜
    ⊢ Asymptotics.IsLittleOTVS 𝕜 l (fun x => HSMul.hSMul (c x) (f x)) fun x => HSM …
  -/
  unfold IsLittleOTVS at *
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommGroup F
    inst✝¹ : TopologicalSpace F
    inst✝ : Module 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    h : ∀ (U : Set E), Membership.mem (nhds 0) U → Exists fun V => And (Membership …
    c : α → 𝕜
    ⊢ ∀ (U : Set E), Membership.mem (nhds 0) U → Exists fun V => And (Membership.m …
  -/
  peel h with U hU V hV ε hε x hx
  /-
    case h.h.h.h.h.h.hq
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommGroup F
    inst✝¹ : TopologicalSpace F
    inst✝ : Module 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    h : ∀ (U : Set E), Membership.mem (nhds 0) U → Exists fun V => And (Membership …
    c : α → 𝕜
    U : Set E
    hU : Membership.mem (nhds 0) U
    V : Set F
    hV : Membership.mem (nhds 0) V
    ε : NNReal
    hε : Ne ε 0
    x : α
    hx : LE.le (egauge 𝕜 U (f x)) (HMul.hMul (↑ε) (egauge 𝕜 V (g x)))
    ⊢ LE.le (egauge 𝕜 U ((fun x => HSMul.hSMul (c x) (f x)) x)) (HMul.hMul (↑ε) (e …
  -/
  rw [egauge_smul_right, egauge_smul_right, mul_left_comm]
    /-
      case h.h.h.h.h.h.hq
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalSpace F
      inst✝ : Module 𝕜 F
      f : α → E
      g : α → F
      l : Filter α
      h : ∀ (U : Set E), Membership.mem (nhds 0) U → Exists fun V => And (Membership …
      c : α → 𝕜
      U : Set E
      hU : Membership.mem (nhds 0) U
      V : Set F
      hV : Membership.mem (nhds 0) V
      ε : NNReal
      hε : Ne ε 0
      x : α
      hx : LE.le (egauge 𝕜 U (f x)) (HMul.hMul (↑ε) (egauge 𝕜 V (g x)))
      ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm (c x))) (egauge 𝕜 U (f x))) (HMul.hMul (↑( …
    -/
  · gcongr
    /-
      🎉 no goals
    -/
  /-
    case h.h.h.h.h.h.hq.h
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommGroup F
    inst✝¹ : TopologicalSpace F
    inst✝ : Module 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    h : ∀ (U : Set E), Membership.mem (nhds 0) U → Exists fun V => And (Membership …
    c : α → 𝕜
    U : Set E
    hU : Membership.mem (nhds 0) U
    V : Set F
    hV : Membership.mem (nhds 0) V
    ε : NNReal
    hε : Ne ε 0
    x : α
    hx : LE.le (egauge 𝕜 U (f x)) (HMul.hMul (↑ε) (egauge 𝕜 V (g x)))
    ⊢ Eq (c x) 0 → V.Nonempty
  -/
  all_goals exact fun _ ↦ Filter.nonempty_of_mem ‹_›
  /-
    🎉 no goals
  -/


lemma isLittleOTVS_one [ContinuousSMul 𝕜 E] {f : α → E} {l : Filter α} :
    f =o[𝕜;l] (1 : α → 𝕜) ↔ Tendsto f l (𝓝 0) := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → E
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 l f 1) (Filter.Tendsto f l (nhds 0))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      ⊢ Asymptotics.IsLittleOTVS 𝕜 l f 1 → Filter.Tendsto f l (nhds 0)
    -/
  · intro hf
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : Asymptotics.IsLittleOTVS 𝕜 l f 1
      ⊢ Filter.Tendsto f l (nhds 0)
    -/
    rw [(basis_sets _).isLittleOTVS_iff nhds_basis_ball] at hf
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      ⊢ Filter.Tendsto f l (nhds 0)
    -/
    rw [(nhds_basis_balanced 𝕜 E).tendsto_right_iff]
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      ⊢ ∀ (i : Set E), And (Membership.mem (nhds 0) i) (Balanced 𝕜 i) → Filter.Event …
    -/
    rintro U ⟨hU, hUb⟩
    /-
      case mp.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      U : Set E
      hU : Membership.mem (nhds 0) U
      hUb : Balanced 𝕜 U
      ⊢ Filter.Eventually (fun x => Membership.mem (id U) (f x)) l
    -/
    rcases hf U hU with ⟨r, hr₀, hr⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      U : Set E
      hU : Membership.mem (nhds 0) U
      hUb : Balanced 𝕜 U
      r : Real
      hr₀ : LT.lt 0 r
      hr : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 (id  …
      ⊢ Filter.Eventually (fun x => Membership.mem (id U) (f x)) l
    -/
    lift r to ℝ≥0 using hr₀.le
    /-
      case mp.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      U : Set E
      hU : Membership.mem (nhds 0) U
      hUb : Balanced 𝕜 U
      r : NNReal
      hr₀ : LT.lt 0 ↑r
      hr : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 (id  …
      ⊢ Filter.Eventually (fun x => Membership.mem (id U) (f x)) l
    -/
    norm_cast at hr₀
    /-
      case mp.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      U : Set E
      hU : Membership.mem (nhds 0) U
      hUb : Balanced 𝕜 U
      r : NNReal
      hr : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 (id  …
      hr₀ : LT.lt 0 r
      ⊢ Filter.Eventually (fun x => Membership.mem (id U) (f x)) l
    -/
    rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
    obtain ⟨ε, hε₀, hε⟩ : ∃ ε : ℝ≥0, 0 < ε ∧ (ε * ‖c‖₊ / r : ℝ≥0∞) < 1 := by
      apply Eventually.exists_gt
      refine Continuous.tendsto' ?_ _ _ (by simp) |>.eventually_lt_const zero_lt_one
      fun_prop (disch := intros; first | apply ENNReal.coe_ne_top | positivity)
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      U : Set E
      hU : Membership.mem (nhds 0) U
      hUb : Balanced 𝕜 U
      r : NNReal
      hr : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 (id  …
      hr₀ : LT.lt 0 r
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : NNReal
      hε₀ : LT.lt 0 ε
      hε : LT.lt (HDiv.hDiv (HMul.hMul ↑ε ↑(NNNorm.nnnorm c)) ↑r) 1
      ⊢ Filter.Eventually (fun x => Membership.mem (id U) (f x)) l
    -/
    filter_upwards [hr ε hε₀.ne'] with x hx
    /-
      case h
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j …
      U : Set E
      hU : Membership.mem (nhds 0) U
      hUb : Balanced 𝕜 U
      r : NNReal
      hr : ∀ (ε : NNReal), Ne ε 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜 (id  …
      hr₀ : LT.lt 0 r
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : NNReal
      hε₀ : LT.lt 0 ε
      hε : LT.lt (HDiv.hDiv (HMul.hMul ↑ε ↑(NNNorm.nnnorm c)) ↑r) 1
      x : α
      hx : LE.le (egauge 𝕜 (id U) (f x)) (HMul.hMul (↑ε) (egauge 𝕜 (Metric.ball 0 ↑r …
      ⊢ Membership.mem (id U) (f x)
    -/
    refine mem_of_egauge_lt_one hUb (hx.trans_lt ?_)
    calc
      (ε : ℝ≥0∞) * egauge 𝕜 (ball (0 : 𝕜) r) 1 ≤ (ε * ‖c‖₊ / r : ℝ≥0∞) := by
        rw [mul_div_assoc]
        gcongr
        simpa using egauge_ball_le_of_one_lt_norm (r := r) (x := (1 : 𝕜)) hc (by simp)
      _ < 1 := ‹_›
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      ⊢ Filter.Tendsto f l (nhds 0) → Asymptotics.IsLittleOTVS 𝕜 l f 1
    -/
  · intro hf U hU
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : Filter.Tendsto f l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (∀ (ε : NNReal), Ne ε 0 → Fi …
    -/
    refine ⟨ball 0 1, ball_mem_nhds _ one_pos, fun ε hε ↦ ?_⟩
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : Filter.Tendsto f l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      ⊢ Filter.Eventually (fun x => LE.le (egauge 𝕜 U (f x)) (HMul.hMul (↑ε) (egauge …
    -/
    rcases NormedField.exists_norm_lt 𝕜 hε.bot_lt with ⟨c, hc₀, hcε⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : Filter.Tendsto f l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : LT.lt 0 (Norm.norm c)
      hcε : LT.lt (Norm.norm c) ((fun a => ↑a) ε)
      ⊢ Filter.Eventually (fun x => LE.le (egauge 𝕜 U (f x)) (HMul.hMul (↑ε) (egauge …
    -/
    replace hc₀ : c ≠ 0 := by simpa using hc₀
    /-
      case mpr.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → E
      l : Filter α
      hf : Filter.Tendsto f l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hcε : LT.lt (Norm.norm c) ((fun a => ↑a) ε)
      hc₀ : Ne c 0
      ⊢ Filter.Eventually (fun x => LE.le (egauge 𝕜 U (f x)) (HMul.hMul (↑ε) (egauge …
    -/
    filter_upwards [hf ((set_smul_mem_nhds_zero_iff hc₀).2 hU)] with a ha
    calc
      egauge 𝕜 U (f a) ≤ ‖c‖₊ := egauge_le_of_mem_smul ha
      _ ≤ ε := mod_cast hcε.le
      _ ≤ ε * egauge 𝕜 (ball (0 : 𝕜) 1) 1 := by
        apply le_mul_of_one_le_right'
        simpa using le_egauge_ball_one 𝕜 (1 : 𝕜)


lemma IsLittleOTVS.tendsto_inv_smul [ContinuousSMul 𝕜 E] {f : α → 𝕜} {g : α → E} {l : Filter α}
    (h : g =o[𝕜;l] f) : Tendsto (fun x ↦ (f x)⁻¹ • g x) l (𝓝 0) := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : Asymptotics.IsLittleOTVS 𝕜 l g f
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
  -/
  rw [(basis_sets _).isLittleOTVS_iff nhds_basis_ball] at h
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
  -/
  rw [(nhds_basis_balanced 𝕜 E).tendsto_right_iff]
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    ⊢ ∀ (i : Set E), And (Membership.mem (nhds 0) i) (Balanced 𝕜 i) → Filter.Event …
  -/
  rintro U ⟨hU, hUB⟩
  /-
    case intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    U : Set E
    hU : Membership.mem (nhds 0) U
    hUB : Balanced 𝕜 U
    ⊢ Filter.Eventually (fun x => Membership.mem (id U) (HSMul.hSMul (Inv.inv (f x …
  -/
  rcases h U hU with ⟨ε, hε₀, hε⟩
  /-
    case intro.intro.intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    U : Set E
    hU : Membership.mem (nhds 0) U
    hUB : Balanced 𝕜 U
    ε : Real
    hε₀ : LT.lt 0 ε
    hε : ∀ (ε_1 : NNReal), Ne ε_1 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜  …
    ⊢ Filter.Eventually (fun x => Membership.mem (id U) (HSMul.hSMul (Inv.inv (f x …
  -/
  lift ε to ℝ≥0 using hε₀.le; norm_cast at hε₀
  /-
    case intro.intro.intro.intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    U : Set E
    hU : Membership.mem (nhds 0) U
    hUB : Balanced 𝕜 U
    ε : NNReal
    hε : ∀ (ε_1 : NNReal), Ne ε_1 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜  …
    hε₀ : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => Membership.mem (id U) (HSMul.hSMul (Inv.inv (f x …
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  filter_upwards [hε (ε / 2 / ‖c‖₊) (ne_of_gt <| div_pos (half_pos hε₀) (one_pos.trans hc))]
    with x hx
  /-
    case h
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    U : Set E
    hU : Membership.mem (nhds 0) U
    hUB : Balanced 𝕜 U
    ε : NNReal
    hε : ∀ (ε_1 : NNReal), Ne ε_1 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜  …
    hε₀ : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : α
    hx : LE.le (egauge 𝕜 (id U) (g x)) (HMul.hMul (↑(HDiv.hDiv (HDiv.hDiv ε 2) (NN …
    ⊢ Membership.mem (id U) (HSMul.hSMul (Inv.inv (f x)) (g x))
  -/
  refine mem_of_egauge_lt_one hUB ?_
  /-
    case h
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h : ∀ (i : Set E), Membership.mem (nhds 0) i → Exists fun j => And (LT.lt 0 j) …
    U : Set E
    hU : Membership.mem (nhds 0) U
    hUB : Balanced 𝕜 U
    ε : NNReal
    hε : ∀ (ε_1 : NNReal), Ne ε_1 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜  …
    hε₀ : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : α
    hx : LE.le (egauge 𝕜 (id U) (g x)) (HMul.hMul (↑(HDiv.hDiv (HDiv.hDiv ε 2) (NN …
    ⊢ LT.lt (egauge 𝕜 (id U) (HSMul.hSMul (Inv.inv (f x)) (g x))) 1
  -/
  rw [id, egauge_smul_right (fun _ ↦ Filter.nonempty_of_mem hU), nnnorm_inv]
  calc
    ↑‖f x‖₊⁻¹ * egauge 𝕜 U (g x)
      ≤ (↑‖f x‖₊)⁻¹ * (↑(ε / 2 / ‖c‖₊) * egauge 𝕜 (ball 0 ε) (f x)) :=
      mul_le_mul' ENNReal.coe_inv_le hx
    _ ≤ (↑‖f x‖₊)⁻¹ * ((ε / 2 / ‖c‖₊) * (‖c‖₊ * ‖f x‖₊ / ε)) := by
      gcongr
      · refine ENNReal.coe_div_le.trans ?_; gcongr; apply ENNReal.coe_div_le
      · exact egauge_ball_le_of_one_lt_norm hc (.inl hε₀.ne')
    _ = (‖f x‖₊ / ‖f x‖₊) * (ε / ε) * (‖c‖₊ / ‖c‖₊) * (1 / 2) := by
      simp only [div_eq_mul_inv, one_mul]; ring
    _ ≤ 1 * 1 * 1 * (1 / 2) := by gcongr <;> apply ENNReal.div_self_le_one
    _ < 1 := by norm_num


lemma isLittleOTVS_iff_tendsto_inv_smul [ContinuousSMul 𝕜 E] {f : α → 𝕜} {g : α → E} {l : Filter α}
    (h₀ : ∀ᶠ x in l, f x = 0 → g x = 0) :
    g =o[𝕜;l] f ↔ Tendsto (fun x ↦ (f x)⁻¹ • g x) l (𝓝 0) := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 l g f) (Filter.Tendsto (fun x => HSMul.hSMul …
  -/
  refine ⟨IsLittleOTVS.tendsto_inv_smul, fun h U hU ↦ ?_⟩
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
    U : Set E
    hU : Membership.mem (nhds 0) U
    ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (∀ (ε : NNReal), Ne ε 0 → Fi …
  -/
  refine ⟨ball 0 1, ball_mem_nhds _ one_pos, fun ε hε ↦ ?_⟩
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
    U : Set E
    hU : Membership.mem (nhds 0) U
    ε : NNReal
    hε : Ne ε 0
    ⊢ Filter.Eventually (fun x => LE.le (egauge 𝕜 U (g x)) (HMul.hMul (↑ε) (egauge …
  -/
  rcases NormedField.exists_norm_lt 𝕜 hε.bot_lt with ⟨c, hc₀, hcε : ‖c‖₊ < ε⟩
  /-
    case intro.intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
    U : Set E
    hU : Membership.mem (nhds 0) U
    ε : NNReal
    hε : Ne ε 0
    c : 𝕜
    hc₀ : LT.lt 0 (Norm.norm c)
    hcε : LT.lt (NNNorm.nnnorm c) ε
    ⊢ Filter.Eventually (fun x => LE.le (egauge 𝕜 U (g x)) (HMul.hMul (↑ε) (egauge …
  -/
  rw [norm_pos_iff] at hc₀
  filter_upwards [h₀, h <| (set_smul_mem_nhds_zero_iff hc₀).2 hU]
    with x hx₀ (hx : (f x)⁻¹ • g x ∈ c • U)
  /-
    case h
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalSpace E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    f : α → 𝕜
    g : α → E
    l : Filter α
    h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
    U : Set E
    hU : Membership.mem (nhds 0) U
    ε : NNReal
    hε : Ne ε 0
    c : 𝕜
    hc₀ : Ne c 0
    hcε : LT.lt (NNNorm.nnnorm c) ε
    x : α
    hx₀ : Eq (f x) 0 → Eq (g x) 0
    hx : Membership.mem (HSMul.hSMul c U) (HSMul.hSMul (Inv.inv (f x)) (g x))
    ⊢ LE.le (egauge 𝕜 U (g x)) (HMul.hMul (↑ε) (egauge 𝕜 (Metric.ball 0 1) (f x)))
  -/
  rcases eq_or_ne (f x) 0 with hf₀ | hf₀
    /-
      case h.inl
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → 𝕜
      g : α → E
      l : Filter α
      h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : Ne c 0
      hcε : LT.lt (NNNorm.nnnorm c) ε
      x : α
      hx₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Membership.mem (HSMul.hSMul c U) (HSMul.hSMul (Inv.inv (f x)) (g x))
      hf₀ : Eq (f x) 0
      ⊢ LE.le (egauge 𝕜 U (g x)) (HMul.hMul (↑ε) (egauge 𝕜 (Metric.ball 0 1) (f x)))
    -/
  · simp [hx₀ hf₀, Filter.nonempty_of_mem hU]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → 𝕜
      g : α → E
      l : Filter α
      h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : Ne c 0
      hcε : LT.lt (NNNorm.nnnorm c) ε
      x : α
      hx₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Membership.mem (HSMul.hSMul c U) (HSMul.hSMul (Inv.inv (f x)) (g x))
      hf₀ : Ne (f x) 0
      ⊢ LE.le (egauge 𝕜 U (g x)) (HMul.hMul (↑ε) (egauge 𝕜 (Metric.ball 0 1) (f x)))
    -/
  · rw [mem_smul_set_iff_inv_smul_mem₀ hc₀, smul_smul] at hx
    /-
      case h.inr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → 𝕜
      g : α → E
      l : Filter α
      h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : Ne c 0
      hcε : LT.lt (NNNorm.nnnorm c) ε
      x : α
      hx₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Membership.mem U (HSMul.hSMul (HMul.hMul (Inv.inv c) (Inv.inv (f x))) (g  …
      hf₀ : Ne (f x) 0
      ⊢ LE.le (egauge 𝕜 U (g x)) (HMul.hMul (↑ε) (egauge 𝕜 (Metric.ball 0 1) (f x)))
    -/
    refine (egauge_le_of_smul_mem_of_ne hx (by simp [*])).trans ?_
    /-
      case h.inr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → 𝕜
      g : α → E
      l : Filter α
      h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : Ne c 0
      hcε : LT.lt (NNNorm.nnnorm c) ε
      x : α
      hx₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Membership.mem U (HSMul.hSMul (HMul.hMul (Inv.inv c) (Inv.inv (f x))) (g  …
      hf₀ : Ne (f x) 0
      ⊢ LE.le (↑(Inv.inv (NNNorm.nnnorm (HMul.hMul (Inv.inv c) (Inv.inv (f x)))))) ( …
    -/
    simp_rw [nnnorm_mul, nnnorm_inv, mul_inv, inv_inv, ENNReal.coe_mul]
    /-
      case h.inr
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → 𝕜
      g : α → E
      l : Filter α
      h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : Ne c 0
      hcε : LT.lt (NNNorm.nnnorm c) ε
      x : α
      hx₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Membership.mem U (HSMul.hSMul (HMul.hMul (Inv.inv c) (Inv.inv (f x))) (g  …
      hf₀ : Ne (f x) 0
      ⊢ LE.le (HMul.hMul ↑(NNNorm.nnnorm c) ↑(NNNorm.nnnorm (f x))) (HMul.hMul (↑ε)  …
    -/
    gcongr
    /-
      case h.inr.h₂
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalSpace E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousSMul 𝕜 E
      f : α → 𝕜
      g : α → E
      l : Filter α
      h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      h : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (f x)) (g x)) l (nhds 0)
      U : Set E
      hU : Membership.mem (nhds 0) U
      ε : NNReal
      hε : Ne ε 0
      c : 𝕜
      hc₀ : Ne c 0
      hcε : LT.lt (NNNorm.nnnorm c) ε
      x : α
      hx₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Membership.mem U (HSMul.hSMul (HMul.hMul (Inv.inv c) (Inv.inv (f x))) (g  …
      hf₀ : Ne (f x) 0
      ⊢ LE.le (↑(NNNorm.nnnorm (f x))) (egauge 𝕜 (Metric.ball 0 1) (f x))
    -/
    apply le_egauge_ball_one
    /-
      🎉 no goals
    -/


lemma isLittleOTVS_iff_isLittleO {f : α → E} {g : α → F} {l : Filter α} :
    f =o[𝕜;l] g ↔ f =o[l] g := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 l f g) (Asymptotics.IsLittleO l f g)
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc : 1 < ‖c‖₊⟩
  /-
    case intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    c : 𝕜
    hc : LT.lt 1 (NNNorm.nnnorm c)
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 l f g) (Asymptotics.IsLittleO l f g)
  -/
  have hc₀ : 0 < ‖c‖₊ := one_pos.trans hc
  /-
    case intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    c : 𝕜
    hc : LT.lt 1 (NNNorm.nnnorm c)
    hc₀ : LT.lt 0 (NNNorm.nnnorm c)
    ⊢ Iff (Asymptotics.IsLittleOTVS 𝕜 l f g) (Asymptotics.IsLittleO l f g)
  -/
  simp only [isLittleO_iff, nhds_basis_ball.isLittleOTVS_iff nhds_basis_ball]
  /-
    case intro
    α : Type u_1
    𝕜 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    f : α → E
    g : α → F
    l : Filter α
    c : 𝕜
    hc : LT.lt 1 (NNNorm.nnnorm c)
    hc₀ : LT.lt 0 (NNNorm.nnnorm c)
    ⊢ Iff (∀ (i : Real), LT.lt 0 i → Exists fun j => And (LT.lt 0 j) (∀ (ε : NNRea …
  -/
  refine ⟨fun h ε hε ↦ ?_, fun h ε hε ↦ ⟨1, one_pos, fun δ hδ ↦ ?_⟩⟩
    /-
      case intro.refine_1
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      f : α → E
      g : α → F
      l : Filter α
      c : 𝕜
      hc : LT.lt 1 (NNNorm.nnnorm c)
      hc₀ : LT.lt 0 (NNNorm.nnnorm c)
      h : ∀ (i : Real), LT.lt 0 i → Exists fun j => And (LT.lt 0 j) (∀ (ε : NNReal), …
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul ε (Norm.norm  …
    -/
  · rcases h ε hε with ⟨δ, hδ₀, hδ⟩
    /-
      case intro.refine_1.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      f : α → E
      g : α → F
      l : Filter α
      c : 𝕜
      hc : LT.lt 1 (NNNorm.nnnorm c)
      hc₀ : LT.lt 0 (NNNorm.nnnorm c)
      h : ∀ (i : Real), LT.lt 0 i → Exists fun j => And (LT.lt 0 j) (∀ (ε : NNReal), …
      ε : Real
      hε : LT.lt 0 ε
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : ∀ (ε_1 : NNReal), Ne ε_1 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜  …
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul ε (Norm.norm  …
    -/
    lift ε to ℝ≥0 using hε.le; lift δ to ℝ≥0 using hδ₀.le; norm_cast at hε hδ₀
    /-
      case intro.refine_1.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      f : α → E
      g : α → F
      l : Filter α
      c : 𝕜
      hc : LT.lt 1 (NNNorm.nnnorm c)
      hc₀ : LT.lt 0 (NNNorm.nnnorm c)
      h : ∀ (i : Real), LT.lt 0 i → Exists fun j => And (LT.lt 0 j) (∀ (ε : NNReal), …
      ε δ : NNReal
      hδ : ∀ (ε_1 : NNReal), Ne ε_1 0 → Filter.Eventually (fun x => LE.le (egauge 𝕜  …
      hε : LT.lt 0 ε
      hδ₀ : LT.lt 0 δ
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul (↑ε) (Norm.no …
    -/
    filter_upwards [hδ (δ / ‖c‖₊) (div_pos hδ₀ hc₀).ne'] with x hx
    suffices (‖f x‖₊ / ε : ℝ≥0∞) ≤ ‖g x‖₊ by
      rw [← ENNReal.coe_div hε.ne'] at this
      rw [← div_le_iff₀' (NNReal.coe_pos.2 hε)]
      exact_mod_cast this
    calc
      (‖f x‖₊ / ε : ℝ≥0∞) ≤ egauge 𝕜 (ball 0 ε) (f x) := div_le_egauge_ball 𝕜 _ _
      _ ≤ ↑(δ / ‖c‖₊) * egauge 𝕜 (ball 0 ↑δ) (g x) := hx
      _ ≤ (δ / ‖c‖₊) * (‖c‖₊ * ‖g x‖₊ / δ) := by
        gcongr
        exacts [ENNReal.coe_div_le, egauge_ball_le_of_one_lt_norm hc (.inl <| ne_of_gt hδ₀)]
      _ = (δ / δ) * (‖c‖₊ / ‖c‖₊) * ‖g x‖₊ := by simp only [div_eq_mul_inv]; ring
      _ ≤ 1 * 1 * ‖g x‖₊ := by gcongr <;> exact ENNReal.div_self_le_one
      _ = ‖g x‖₊ := by simp
    /-
      case intro.refine_2
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      f : α → E
      g : α → F
      l : Filter α
      c : 𝕜
      hc : LT.lt 1 (NNNorm.nnnorm c)
      hc₀ : LT.lt 0 (NNNorm.nnnorm c)
      h : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f  …
      ε : Real
      hε : LT.lt 0 ε
      δ : NNReal
      hδ : Ne δ 0
      ⊢ Filter.Eventually (fun x => LE.le (egauge 𝕜 (Metric.ball 0 ε) (f x)) (HMul.h …
    -/
  · filter_upwards [@h ↑(ε * δ / ‖c‖₊) (by positivity)] with x (hx : ‖f x‖₊ ≤ ε * δ / ‖c‖₊ * ‖g x‖₊)
    /-
      case h
      α : Type u_1
      𝕜 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      f : α → E
      g : α → F
      l : Filter α
      c : 𝕜
      hc : LT.lt 1 (NNNorm.nnnorm c)
      hc₀ : LT.lt 0 (NNNorm.nnnorm c)
      h : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f  …
      ε : Real
      hε : LT.lt 0 ε
      δ : NNReal
      hδ : Ne δ 0
      x : α
      hx : LE.le (↑(NNNorm.nnnorm (f x))) (HMul.hMul (HDiv.hDiv (HMul.hMul ε ↑δ) ↑(N …
      ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ε) (f x)) (HMul.hMul (↑δ) (egauge 𝕜 (Metric.b …
    -/
    lift ε to ℝ≥0 using hε.le
    calc
      egauge 𝕜 (ball 0 ε) (f x) ≤ ‖c‖₊ * ‖f x‖₊ / ε :=
        egauge_ball_le_of_one_lt_norm hc (.inl <| ne_of_gt hε)
      _ ≤ ‖c‖₊ * (↑(ε * δ / ‖c‖₊) * ‖g x‖₊) / ε := by gcongr; exact_mod_cast hx
      _ = (‖c‖₊ / ‖c‖₊) * (ε / ε) * δ * ‖g x‖₊ := by
        simp only [div_eq_mul_inv, ENNReal.coe_inv hc₀.ne', ENNReal.coe_mul]; ring
      _ ≤ 1 * 1 * δ * ‖g x‖₊ := by gcongr <;> exact ENNReal.div_self_le_one
      _ = δ * ‖g x‖₊ := by simp
      _ ≤ δ * egauge 𝕜 (ball 0 1) (g x) := by gcongr; apply le_egauge_ball_one


alias ⟨isLittleOTVS.isLittleO, IsLittle.isLittleOTVS⟩ := isLittleOTVS_iff_isLittleO


