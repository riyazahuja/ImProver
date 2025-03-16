/-- A complete nontrivially normed field has cardinality at least continuum. -/
theorem continuum_le_cardinal_of_nontriviallyNormedField
    (𝕜 : Type*) [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜] : 𝔠 ≤ #𝕜 := by
  suffices ∃ f : (ℕ → Bool) → 𝕜, range f ⊆ univ ∧ Continuous f ∧ Injective f by
    rcases this with ⟨f, -, -, f_inj⟩
    simpa using lift_mk_le_lift_mk_of_injective f_inj
  /-
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) Set.univ) (And (Continuo …
  -/
  apply Perfect.exists_nat_bool_injection _ univ_nonempty
  /-
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    ⊢ Perfect Set.univ
  -/
  refine ⟨isClosed_univ, preperfect_iff_nhds.2 (fun x _ U hU ↦ ?_)⟩
  /-
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    ⊢ Exists fun y => And (Membership.mem (Inter.inter U Set.univ) y) (Ne y x)
  -/
  rcases NormedField.exists_norm_lt_one 𝕜 with ⟨c, c_pos, hc⟩
  have A : Tendsto (fun n ↦ x + c^n) atTop (𝓝 (x + 0)) :=
    tendsto_const_nhds.add (tendsto_pow_atTop_nhds_zero_of_norm_lt_one hc)
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds ( …
    ⊢ Exists fun y => And (Membership.mem (Inter.inter U Set.univ) y) (Ne y x)
  -/
  rw [add_zero] at A
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds x)
    ⊢ Exists fun y => And (Membership.mem (Inter.inter U Set.univ) y) (Ne y x)
  -/
  have B : ∀ᶠ n in atTop, x + c^n ∈ U := tendsto_def.1 A U hU
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds x)
    B : Filter.Eventually (fun n => Membership.mem U (HAdd.hAdd x (HPow.hPow c n)) …
    ⊢ Exists fun y => And (Membership.mem (Inter.inter U Set.univ) y) (Ne y x)
  -/
  rcases B.exists with ⟨n, hn⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds x)
    B : Filter.Eventually (fun n => Membership.mem U (HAdd.hAdd x (HPow.hPow c n)) …
    n : Nat
    hn : Membership.mem U (HAdd.hAdd x (HPow.hPow c n))
    ⊢ Exists fun y => And (Membership.mem (Inter.inter U Set.univ) y) (Ne y x)
  -/
  refine ⟨x + c^n, by simpa using hn, ?_⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds x)
    B : Filter.Eventually (fun n => Membership.mem U (HAdd.hAdd x (HPow.hPow c n)) …
    n : Nat
    hn : Membership.mem U (HAdd.hAdd x (HPow.hPow c n))
    ⊢ Ne (HAdd.hAdd x (HPow.hPow c n)) x
  -/
  simp only [ne_eq, add_right_eq_self]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds x)
    B : Filter.Eventually (fun n => Membership.mem U (HAdd.hAdd x (HPow.hPow c n)) …
    n : Nat
    hn : Membership.mem U (HAdd.hAdd x (HPow.hPow c n))
    ⊢ Not (Eq (HPow.hPow c n) 0)
  -/
  apply pow_ne_zero
  /-
    case intro.intro.intro.h
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : CompleteSpace 𝕜
    x : 𝕜
    x✝ : Membership.mem Set.univ x
    U : Set 𝕜
    hU : Membership.mem (nhds x) U
    c : 𝕜
    c_pos : LT.lt 0 (Norm.norm c)
    hc : LT.lt (Norm.norm c) 1
    A : Filter.Tendsto (fun n => HAdd.hAdd x (HPow.hPow c n)) Filter.atTop (nhds x)
    B : Filter.Eventually (fun n => Membership.mem U (HAdd.hAdd x (HPow.hPow c n)) …
    n : Nat
    hn : Membership.mem U (HAdd.hAdd x (HPow.hPow c n))
    ⊢ Ne c 0
  -/
  simpa using c_pos
  /-
    🎉 no goals
  -/


/-- A nontrivial module over a complete nontrivially normed field has cardinality at least
continuum. -/
theorem continuum_le_cardinal_of_module
    (𝕜 : Type u) (E : Type v) [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜]
    [AddCommGroup E] [Module 𝕜 E] [Nontrivial E] : 𝔠 ≤ #E := by
  have A : lift.{v} (𝔠 : Cardinal.{u}) ≤ lift.{v} (#𝕜) := by
    simpa using continuum_le_cardinal_of_nontriviallyNormedField 𝕜
  /-
    𝕜 : Type u
    E : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : CompleteSpace 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nontrivial E
    A : LE.le (Cardinal.lift.{v, u} Cardinal.continuum) (Cardinal.lift.{v, u} (Car …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk E)
  -/
  simpa using A.trans (Cardinal.mk_le_of_module 𝕜 E)
  /-
    🎉 no goals
  -/


/-- In a topological vector space over a nontrivially normed field, any neighborhood of zero has
the same cardinality as the whole space.

See also `cardinal_eq_of_mem_nhds`. -/
lemma cardinal_eq_of_mem_nhds_zero
    {E : Type*} (𝕜 : Type*) [NontriviallyNormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    [TopologicalSpace E] [ContinuousSMul 𝕜 E] {s : Set E} (hs : s ∈ 𝓝 (0 : E)) : #s = #E := by
  /- As `s` is a neighborhood of `0`, the space is covered by the rescaled sets `c^n • s`,
  where `c` is any element of `𝕜` with norm `> 1`. All these sets are in bijection and have
  therefore the same cardinality. The conclusion follows. -/
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Membership.mem (nhds 0) s
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  obtain ⟨c, hc⟩ : ∃ x : 𝕜 , 1 < ‖x‖ := NormedField.exists_lt_norm 𝕜 1
  have cn_ne : ∀ n, c^n ≠ 0 := by
    intro n
    apply pow_ne_zero
    rintro rfl
    simp only [norm_zero] at hc
    exact lt_irrefl _ (hc.trans zero_lt_one)
  have A : ∀ (x : E), ∀ᶠ n in (atTop : Filter ℕ), x ∈ c^n • s := by
    intro x
    have : Tendsto (fun n ↦ (c^n) ⁻¹ • x) atTop (𝓝 ((0 : 𝕜) • x)) := by
      have : Tendsto (fun n ↦ (c^n)⁻¹) atTop (𝓝 0) := by
        simp_rw [← inv_pow]
        apply tendsto_pow_atTop_nhds_zero_of_norm_lt_one
        rw [norm_inv]
        exact inv_lt_one_of_one_lt₀ hc
      exact Tendsto.smul_const this x
    rw [zero_smul] at this
    filter_upwards [this hs] with n (hn : (c ^ n)⁻¹ • x ∈ s)
    exact (mem_smul_set_iff_inv_smul_mem₀ (cn_ne n) _ _).2 hn
  have B : ∀ n, #(c^n • s :) = #s := by
    intro n
    have : (c^n • s :) ≃ s :=
    { toFun := fun x ↦ ⟨(c^n)⁻¹ • x.1, (mem_smul_set_iff_inv_smul_mem₀ (cn_ne n) _ _).1 x.2⟩
      invFun := fun x ↦ ⟨(c^n) • x.1, smul_mem_smul_set x.2⟩
      left_inv := fun x ↦ by simp [smul_smul, mul_inv_cancel₀ (cn_ne n)]
      right_inv := fun x ↦ by simp [smul_smul, inv_mul_cancel₀ (cn_ne n)] }
    exact Cardinal.mk_congr this
  /-
    case intro
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Membership.mem (nhds 0) s
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    cn_ne : ∀ (n : Nat), Ne (HPow.hPow c n) 0
    A : ∀ (x : E), Filter.Eventually (fun n => Membership.mem (HSMul.hSMul (HPow.h …
    B : ∀ (n : Nat), Eq (Cardinal.mk ↑(HSMul.hSMul (HPow.hPow c n) s)) (Cardinal.m …
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  apply (Cardinal.mk_of_countable_eventually_mem A B).symm
  /-
    🎉 no goals
  -/


/-- In a topological vector space over a nontrivially normed field, any neighborhood of a point has
the same cardinality as the whole space. -/
theorem cardinal_eq_of_mem_nhds
    {E : Type*} (𝕜 : Type*) [NontriviallyNormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul 𝕜 E]
    {s : Set E} {x : E} (hs : s ∈ 𝓝 x) : #s = #E := by
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  let g := Homeomorph.addLeft x
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    g : Homeomorph E E := Homeomorph.addLeft x
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  let t := g ⁻¹' s
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    g : Homeomorph E E := Homeomorph.addLeft x
    t : Set E := Set.preimage (⇑g) s
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  have : t ∈ 𝓝 0 := g.continuous.continuousAt.preimage_mem_nhds (by simpa [g] using hs)
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    g : Homeomorph E E := Homeomorph.addLeft x
    t : Set E := Set.preimage (⇑g) s
    this : Membership.mem (nhds 0) t
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  have A : #t = #E := cardinal_eq_of_mem_nhds_zero 𝕜 this
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    g : Homeomorph E E := Homeomorph.addLeft x
    t : Set E := Set.preimage (⇑g) s
    this : Membership.mem (nhds 0) t
    A : Eq (Cardinal.mk ↑t) (Cardinal.mk E)
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  have B : #t = #s := Cardinal.mk_subtype_of_equiv s g.toEquiv
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    x : E
    hs : Membership.mem (nhds x) s
    g : Homeomorph E E := Homeomorph.addLeft x
    t : Set E := Set.preimage (⇑g) s
    this : Membership.mem (nhds 0) t
    A : Eq (Cardinal.mk ↑t) (Cardinal.mk E)
    B : Eq (Cardinal.mk ↑t) (Cardinal.mk ↑s)
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  rwa [B] at A
  /-
    🎉 no goals
  -/


/-- In a topological vector space over a nontrivially normed field, any nonempty open set has
the same cardinality as the whole space. -/
theorem cardinal_eq_of_isOpen
    {E : Type*} (𝕜 : Type*) [NontriviallyNormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul 𝕜 E] {s : Set E}
    (hs : IsOpen s) (h's : s.Nonempty) : #s = #E := by
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  rcases h's with ⟨x, hx⟩
  /-
    case intro
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : IsOpen s
    x : E
    hx : Membership.mem s x
    ⊢ Eq (Cardinal.mk ↑s) (Cardinal.mk E)
  -/
  exact cardinal_eq_of_mem_nhds 𝕜 (hs.mem_nhds hx)
  /-
    🎉 no goals
  -/


/-- In a nontrivial topological vector space over a complete nontrivially normed field, any nonempty
open set has cardinality at least continuum. -/
theorem continuum_le_cardinal_of_isOpen
    {E : Type*} (𝕜 : Type*) [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜] [AddCommGroup E]
    [Module 𝕜 E] [Nontrivial E] [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul 𝕜 E]
    {s : Set E} (hs : IsOpen s) (h's : s.Nonempty) : 𝔠 ≤ #s := by
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : Nontrivial E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : IsOpen s
    h's : s.Nonempty
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑s)
  -/
  simpa [cardinal_eq_of_isOpen 𝕜 hs h's] using continuum_le_cardinal_of_module 𝕜 E
  /-
    🎉 no goals
  -/


/-- In a nontrivial topological vector space over a complete nontrivially normed field, any
countable set has dense complement. -/
theorem Set.Countable.dense_compl
    {E : Type u} (𝕜 : Type*) [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜] [AddCommGroup E]
    [Module 𝕜 E] [Nontrivial E] [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul 𝕜 E]
    {s : Set E} (hs : s.Countable) : Dense sᶜ := by
  /-
    E : Type u
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : Nontrivial E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : s.Countable
    ⊢ Dense (HasCompl.compl s)
  -/
  rw [← interior_eq_empty_iff_dense_compl]
  /-
    E : Type u
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : Nontrivial E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : s.Countable
    ⊢ Eq (interior s) EmptyCollection.emptyCollection
  -/
  by_contra H
  /-
    E : Type u
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : Nontrivial E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : s.Countable
    H : Not (Eq (interior s) EmptyCollection.emptyCollection)
    ⊢ False
  -/
  apply lt_irrefl (ℵ₀ : Cardinal.{u})
  calc
    (ℵ₀ : Cardinal.{u}) < 𝔠 := aleph0_lt_continuum
    _ ≤ #(interior s) :=
      continuum_le_cardinal_of_isOpen 𝕜 isOpen_interior (nmem_singleton_empty.1 H)
    _ ≤ #s := mk_le_mk_of_subset interior_subset
    _ ≤ ℵ₀ := le_aleph0 hs

