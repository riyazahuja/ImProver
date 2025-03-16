theorem Filter.Tendsto.div_const {x : G₀} (hf : Tendsto f l (𝓝 x)) (y : G₀) :
    Tendsto (fun a => f a / y) l (𝓝 (x / y)) := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝² : DivInvMonoid G₀
    inst✝¹ : TopologicalSpace G₀
    inst✝ : ContinuousMul G₀
    f : α → G₀
    l : Filter α
    x : G₀
    hf : Filter.Tendsto f l (nhds x)
    y : G₀
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (f a) y) l (nhds (HDiv.hDiv x y))
  -/
  simpa only [div_eq_mul_inv] using hf.mul tendsto_const_nhds
  /-
    🎉 no goals
  -/


nonrec theorem ContinuousAt.div_const {a : α} (hf : ContinuousAt f a) (y : G₀) :
    ContinuousAt (fun x => f x / y) a :=
  hf.div_const y


nonrec theorem ContinuousWithinAt.div_const {a} (hf : ContinuousWithinAt f s a) (y : G₀) :
    ContinuousWithinAt (fun x => f x / y) s a :=
  hf.div_const _


theorem ContinuousOn.div_const (hf : ContinuousOn f s) (y : G₀) :
    ContinuousOn (fun x => f x / y) s := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝³ : DivInvMonoid G₀
    inst✝² : TopologicalSpace G₀
    inst✝¹ : ContinuousMul G₀
    f : α → G₀
    s : Set α
    inst✝ : TopologicalSpace α
    hf : ContinuousOn f s
    y : G₀
    ⊢ ContinuousOn (fun x => HDiv.hDiv (f x) y) s
  -/
  simpa only [div_eq_mul_inv] using hf.mul continuousOn_const
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
theorem Continuous.div_const (hf : Continuous f) (y : G₀) : Continuous fun x => f x / y := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝³ : DivInvMonoid G₀
    inst✝² : TopologicalSpace G₀
    inst✝¹ : ContinuousMul G₀
    f : α → G₀
    inst✝ : TopologicalSpace α
    hf : Continuous f
    y : G₀
    ⊢ Continuous fun x => HDiv.hDiv (f x) y
  -/
  simpa only [div_eq_mul_inv] using hf.mul continuous_const
  /-
    🎉 no goals
  -/


/-- A type with `0` and `Inv` such that `fun x ↦ x⁻¹` is continuous at all nonzero points. Any
normed (semi)field has this property. -/
class HasContinuousInv₀ (G₀ : Type*) [Zero G₀] [Inv G₀] [TopologicalSpace G₀] : Prop where
  /-- The map `fun x ↦ x⁻¹` is continuous at all nonzero points. -/
  continuousAt_inv₀ : ∀ ⦃x : G₀⦄, x ≠ 0 → ContinuousAt Inv.inv x


theorem tendsto_inv₀ {x : G₀} (hx : x ≠ 0) : Tendsto Inv.inv (𝓝 x) (𝓝 x⁻¹) :=
  continuousAt_inv₀ hx


theorem continuousOn_inv₀ : ContinuousOn (Inv.inv : G₀ → G₀) {0}ᶜ := fun _x hx =>
  (continuousAt_inv₀ hx).continuousWithinAt


/-- If a function converges to a nonzero value, its inverse converges to the inverse of this value.
We use the name `Filter.Tendsto.inv₀` as `Filter.Tendsto.inv` is already used in multiplicative
topological groups. -/
theorem Filter.Tendsto.inv₀ {a : G₀} (hf : Tendsto f l (𝓝 a)) (ha : a ≠ 0) :
    Tendsto (fun x => (f x)⁻¹) l (𝓝 a⁻¹) :=
  (tendsto_inv₀ ha).comp hf


nonrec theorem ContinuousWithinAt.inv₀ (hf : ContinuousWithinAt f s a) (ha : f a ≠ 0) :
    ContinuousWithinAt (fun x => (f x)⁻¹) s a :=
  hf.inv₀ ha


@[fun_prop]
nonrec theorem ContinuousAt.inv₀ (hf : ContinuousAt f a) (ha : f a ≠ 0) :
    ContinuousAt (fun x => (f x)⁻¹) a :=
  hf.inv₀ ha


@[continuity, fun_prop]
theorem Continuous.inv₀ (hf : Continuous f) (h0 : ∀ x, f x ≠ 0) : Continuous fun x => (f x)⁻¹ :=
  continuous_iff_continuousAt.2 fun x => (hf.tendsto x).inv₀ (h0 x)


@[fun_prop]
theorem ContinuousOn.inv₀ (hf : ContinuousOn f s) (h0 : ∀ x ∈ s, f x ≠ 0) :
    ContinuousOn (fun x => (f x)⁻¹) s := fun x hx => (hf x hx).inv₀ (h0 x hx)


/-- If `G₀` is a group with zero with topology such that `x ↦ x⁻¹` is continuous at all nonzero
points. Then the coercion `G₀ˣ → G₀` is a topological embedding. -/
theorem Units.isEmbedding_val₀ [GroupWithZero G₀] [TopologicalSpace G₀] [HasContinuousInv₀ G₀] :
    IsEmbedding (val : G₀ˣ → G₀) :=
  embedding_val_mk <| (continuousOn_inv₀ (G₀ := G₀)).mono fun _ ↦ IsUnit.ne_zero


@[deprecated (since := "2024-10-26")]
alias Units.embedding_val₀ := Units.isEmbedding_val₀


lemma nhds_inv₀ (hx : x ≠ 0) : 𝓝 x⁻¹ = (𝓝 x)⁻¹ := by
  /-
    G₀ : Type u_3
    inst✝² : GroupWithZero G₀
    inst✝¹ : TopologicalSpace G₀
    inst✝ : HasContinuousInv₀ G₀
    x : G₀
    hx : Ne x 0
    ⊢ Eq (nhds (Inv.inv x)) (Inv.inv (nhds x))
  -/
  refine le_antisymm (inv_le_iff_le_inv.1 ?_) (tendsto_inv₀ hx)
  /-
    G₀ : Type u_3
    inst✝² : GroupWithZero G₀
    inst✝¹ : TopologicalSpace G₀
    inst✝ : HasContinuousInv₀ G₀
    x : G₀
    hx : Ne x 0
    ⊢ LE.le (Inv.inv (nhds (Inv.inv x))) (nhds x)
  -/
  simpa only [inv_inv] using tendsto_inv₀ (inv_ne_zero hx)
  /-
    🎉 no goals
  -/


lemma tendsto_inv_iff₀ {l : Filter α} {f : α → G₀} (hx : x ≠ 0) :
    Tendsto (fun x ↦ (f x)⁻¹) l (𝓝 x⁻¹) ↔ Tendsto f l (𝓝 x) := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝² : GroupWithZero G₀
    inst✝¹ : TopologicalSpace G₀
    inst✝ : HasContinuousInv₀ G₀
    x : G₀
    l : Filter α
    f : α → G₀
    hx : Ne x 0
    ⊢ Iff (Filter.Tendsto (fun x => Inv.inv (f x)) l (nhds (Inv.inv x))) (Filter.T …
  -/
  simp only [nhds_inv₀ hx, ← Filter.comap_inv, tendsto_comap_iff, Function.comp_def, inv_inv]
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.div {l : Filter α} {a b : G₀} (hf : Tendsto f l (𝓝 a))
    (hg : Tendsto g l (𝓝 b)) (hy : b ≠ 0) : Tendsto (f / g) l (𝓝 (a / b)) := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝³ : GroupWithZero G₀
    inst✝² : TopologicalSpace G₀
    inst✝¹ : HasContinuousInv₀ G₀
    inst✝ : ContinuousMul G₀
    f g : α → G₀
    l : Filter α
    a b : G₀
    hf : Filter.Tendsto f l (nhds a)
    hg : Filter.Tendsto g l (nhds b)
    hy : Ne b 0
    ⊢ Filter.Tendsto (HDiv.hDiv f g) l (nhds (HDiv.hDiv a b))
  -/
  simpa only [div_eq_mul_inv] using hf.mul (hg.inv₀ hy)
  /-
    🎉 no goals
  -/


theorem Filter.tendsto_mul_iff_of_ne_zero [T1Space G₀] {f g : α → G₀} {l : Filter α} {x y : G₀}
    (hg : Tendsto g l (𝓝 y)) (hy : y ≠ 0) :
    Tendsto (fun n => f n * g n) l (𝓝 <| x * y) ↔ Tendsto f l (𝓝 x) := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : TopologicalSpace G₀
    inst✝² : HasContinuousInv₀ G₀
    inst✝¹ : ContinuousMul G₀
    inst✝ : T1Space G₀
    f g : α → G₀
    l : Filter α
    x y : G₀
    hg : Filter.Tendsto g l (nhds y)
    hy : Ne y 0
    ⊢ Iff (Filter.Tendsto (fun n => HMul.hMul (f n) (g n)) l (nhds (HMul.hMul x y) …
  -/
  refine ⟨fun hfg => ?_, fun hf => hf.mul hg⟩
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : TopologicalSpace G₀
    inst✝² : HasContinuousInv₀ G₀
    inst✝¹ : ContinuousMul G₀
    inst✝ : T1Space G₀
    f g : α → G₀
    l : Filter α
    x y : G₀
    hg : Filter.Tendsto g l (nhds y)
    hy : Ne y 0
    hfg : Filter.Tendsto (fun n => HMul.hMul (f n) (g n)) l (nhds (HMul.hMul x y))
    ⊢ Filter.Tendsto f l (nhds x)
  -/
  rw [← mul_div_cancel_right₀ x hy]
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : TopologicalSpace G₀
    inst✝² : HasContinuousInv₀ G₀
    inst✝¹ : ContinuousMul G₀
    inst✝ : T1Space G₀
    f g : α → G₀
    l : Filter α
    x y : G₀
    hg : Filter.Tendsto g l (nhds y)
    hy : Ne y 0
    hfg : Filter.Tendsto (fun n => HMul.hMul (f n) (g n)) l (nhds (HMul.hMul x y))
    ⊢ Filter.Tendsto f l (nhds (HDiv.hDiv (HMul.hMul x y) y))
  -/
  refine Tendsto.congr' ?_ (hfg.div hg hy)
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : TopologicalSpace G₀
    inst✝² : HasContinuousInv₀ G₀
    inst✝¹ : ContinuousMul G₀
    inst✝ : T1Space G₀
    f g : α → G₀
    l : Filter α
    x y : G₀
    hg : Filter.Tendsto g l (nhds y)
    hy : Ne y 0
    hfg : Filter.Tendsto (fun n => HMul.hMul (f n) (g n)) l (nhds (HMul.hMul x y))
    ⊢ l.EventuallyEq (HDiv.hDiv (fun n => HMul.hMul (f n) (g n)) g) f
  -/
  exact (hg.eventually_ne hy).mono fun n hn => mul_div_cancel_right₀ _ hn
  /-
    🎉 no goals
  -/


nonrec theorem ContinuousWithinAt.div (hf : ContinuousWithinAt f s a)
    (hg : ContinuousWithinAt g s a) (h₀ : g a ≠ 0) : ContinuousWithinAt (f / g) s a :=
  hf.div hg h₀


theorem ContinuousOn.div (hf : ContinuousOn f s) (hg : ContinuousOn g s) (h₀ : ∀ x ∈ s, g x ≠ 0) :
    ContinuousOn (f / g) s := fun x hx => (hf x hx).div (hg x hx) (h₀ x hx)


/-- Continuity at a point of the result of dividing two functions continuous at that point, where
the denominator is nonzero. -/
nonrec theorem ContinuousAt.div (hf : ContinuousAt f a) (hg : ContinuousAt g a) (h₀ : g a ≠ 0) :
    ContinuousAt (f / g) a :=
  hf.div hg h₀


@[continuity]
theorem Continuous.div (hf : Continuous f) (hg : Continuous g) (h₀ : ∀ x, g x ≠ 0) :
                             /-
                               α : Type u_1
                               G₀ : Type u_3
                               inst✝⁴ : GroupWithZero G₀
                               inst✝³ : TopologicalSpace G₀
                               inst✝² : HasContinuousInv₀ G₀
                               inst✝¹ : ContinuousMul G₀
                               f g : α → G₀
                               inst✝ : TopologicalSpace α
                               hf : Continuous f
                               hg : Continuous g
                               h₀ : ∀ (x : α), Ne (g x) 0
                               ⊢ Continuous (HDiv.hDiv f g)
                             -/
    Continuous (f / g) := by simpa only [div_eq_mul_inv] using hf.mul (hg.inv₀ h₀)
                             /-
                               🎉 no goals
                             -/


theorem continuousOn_div : ContinuousOn (fun p : G₀ × G₀ => p.1 / p.2) { p | p.2 ≠ 0 } :=
  continuousOn_fst.div continuousOn_snd fun _ => id


@[fun_prop]
theorem Continuous.div₀ (hf : Continuous f) (hg : Continuous g) (h₀ : ∀ x, g x ≠ 0) :
    Continuous (fun x => f x / g x) := by
  /-
    α : Type u_1
    G₀ : Type u_3
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : TopologicalSpace G₀
    inst✝² : HasContinuousInv₀ G₀
    inst✝¹ : ContinuousMul G₀
    f g : α → G₀
    inst✝ : TopologicalSpace α
    hf : Continuous f
    hg : Continuous g
    h₀ : ∀ (x : α), Ne (g x) 0
    ⊢ Continuous fun x => HDiv.hDiv (f x) (g x)
  -/
  simpa only [div_eq_mul_inv] using hf.mul (hg.inv₀ h₀)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem ContinuousAt.div₀ (hf : ContinuousAt f a) (hg : ContinuousAt g a) (h₀ : g a ≠ 0) :
    ContinuousAt (fun x => f x / g x) a := ContinuousAt.div hf hg h₀


@[fun_prop]
theorem ContinuousOn.div₀ (hf : ContinuousOn f s) (hg : ContinuousOn g s) (h₀ : ∀ x ∈ s, g x ≠ 0) :
    ContinuousOn (fun x => f x / g x) s := ContinuousOn.div hf hg h₀


/-- The function `f x / g x` is discontinuous when `g x = 0`. However, under appropriate
conditions, `h x (f x / g x)` is still continuous.  The condition is that if `g a = 0` then `h x y`
must tend to `h a 0` when `x` tends to `a`, with no information about `y`. This is represented by
the `⊤` filter.  Note: `tendsto_prod_top_iff` characterizes this convergence in uniform spaces.  See
also `Filter.prod_top` and `Filter.mem_prod_top`. -/
theorem ContinuousAt.comp_div_cases {f g : α → G₀} (h : α → G₀ → β) (hf : ContinuousAt f a)
    (hg : ContinuousAt g a) (hh : g a ≠ 0 → ContinuousAt (↿h) (a, f a / g a))
    (h2h : g a = 0 → Tendsto (↿h) (𝓝 a ×ˢ ⊤) (𝓝 (h a 0))) :
    ContinuousAt (fun x => h x (f x / g x)) a := by
  /-
    α : Type u_1
    β : Type u_2
    G₀ : Type u_3
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : TopologicalSpace G₀
    inst✝³ : HasContinuousInv₀ G₀
    inst✝² : ContinuousMul G₀
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a : α
    f g : α → G₀
    h : α → G₀ → β
    hf : ContinuousAt f a
    hg : ContinuousAt g a
    hh : Ne (g a) 0 → ContinuousAt (Function.HasUncurry.uncurry h) { fst := a, snd …
    h2h : Eq (g a) 0 → Filter.Tendsto (Function.HasUncurry.uncurry h) (SProd.sprod …
    ⊢ ContinuousAt (fun x => h x (HDiv.hDiv (f x) (g x))) a
  -/
  show ContinuousAt (↿h ∘ fun x => (x, f x / g x)) a
  /-
    α : Type u_1
    β : Type u_2
    G₀ : Type u_3
    inst✝⁵ : GroupWithZero G₀
    inst✝⁴ : TopologicalSpace G₀
    inst✝³ : HasContinuousInv₀ G₀
    inst✝² : ContinuousMul G₀
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a : α
    f g : α → G₀
    h : α → G₀ → β
    hf : ContinuousAt f a
    hg : ContinuousAt g a
    hh : Ne (g a) 0 → ContinuousAt (Function.HasUncurry.uncurry h) { fst := a, snd …
    h2h : Eq (g a) 0 → Filter.Tendsto (Function.HasUncurry.uncurry h) (SProd.sprod …
    ⊢ ContinuousAt (Function.comp (Function.HasUncurry.uncurry h) fun x => { fst : …
  -/
  by_cases hga : g a = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      G₀ : Type u_3
      inst✝⁵ : GroupWithZero G₀
      inst✝⁴ : TopologicalSpace G₀
      inst✝³ : HasContinuousInv₀ G₀
      inst✝² : ContinuousMul G₀
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      a : α
      f g : α → G₀
      h : α → G₀ → β
      hf : ContinuousAt f a
      hg : ContinuousAt g a
      hh : Ne (g a) 0 → ContinuousAt (Function.HasUncurry.uncurry h) { fst := a, snd …
      h2h : Eq (g a) 0 → Filter.Tendsto (Function.HasUncurry.uncurry h) (SProd.sprod …
      hga : Eq (g a) 0
      ⊢ ContinuousAt (Function.comp (Function.HasUncurry.uncurry h) fun x => { fst : …
    -/
  · rw [ContinuousAt]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      G₀ : Type u_3
      inst✝⁵ : GroupWithZero G₀
      inst✝⁴ : TopologicalSpace G₀
      inst✝³ : HasContinuousInv₀ G₀
      inst✝² : ContinuousMul G₀
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      a : α
      f g : α → G₀
      h : α → G₀ → β
      hf : ContinuousAt f a
      hg : ContinuousAt g a
      hh : Ne (g a) 0 → ContinuousAt (Function.HasUncurry.uncurry h) { fst := a, snd …
      h2h : Eq (g a) 0 → Filter.Tendsto (Function.HasUncurry.uncurry h) (SProd.sprod …
      hga : Eq (g a) 0
      ⊢ Filter.Tendsto (Function.comp (Function.HasUncurry.uncurry h) fun x => { fst …
    -/
    simp_rw [comp_apply, hga, div_zero]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      G₀ : Type u_3
      inst✝⁵ : GroupWithZero G₀
      inst✝⁴ : TopologicalSpace G₀
      inst✝³ : HasContinuousInv₀ G₀
      inst✝² : ContinuousMul G₀
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      a : α
      f g : α → G₀
      h : α → G₀ → β
      hf : ContinuousAt f a
      hg : ContinuousAt g a
      hh : Ne (g a) 0 → ContinuousAt (Function.HasUncurry.uncurry h) { fst := a, snd …
      h2h : Eq (g a) 0 → Filter.Tendsto (Function.HasUncurry.uncurry h) (SProd.sprod …
      hga : Eq (g a) 0
      ⊢ Filter.Tendsto (Function.comp (Function.HasUncurry.uncurry h) fun x => { fst …
    -/
    exact (h2h hga).comp (continuousAt_id.prod_mk tendsto_top)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      G₀ : Type u_3
      inst✝⁵ : GroupWithZero G₀
      inst✝⁴ : TopologicalSpace G₀
      inst✝³ : HasContinuousInv₀ G₀
      inst✝² : ContinuousMul G₀
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      a : α
      f g : α → G₀
      h : α → G₀ → β
      hf : ContinuousAt f a
      hg : ContinuousAt g a
      hh : Ne (g a) 0 → ContinuousAt (Function.HasUncurry.uncurry h) { fst := a, snd …
      h2h : Eq (g a) 0 → Filter.Tendsto (Function.HasUncurry.uncurry h) (SProd.sprod …
      hga : Not (Eq (g a) 0)
      ⊢ ContinuousAt (Function.comp (Function.HasUncurry.uncurry h) fun x => { fst : …
    -/
  · exact ContinuousAt.comp (hh hga) (continuousAt_id.prod (hf.div hg hga))
    /-
      🎉 no goals
    -/


/-- `h x (f x / g x)` is continuous under certain conditions, even if the denominator is sometimes
  `0`. See docstring of `ContinuousAt.comp_div_cases`. -/
theorem Continuous.comp_div_cases {f g : α → G₀} (h : α → G₀ → β) (hf : Continuous f)
    (hg : Continuous g) (hh : ∀ a, g a ≠ 0 → ContinuousAt (↿h) (a, f a / g a))
    (h2h : ∀ a, g a = 0 → Tendsto (↿h) (𝓝 a ×ˢ ⊤) (𝓝 (h a 0))) :
    Continuous fun x => h x (f x / g x) :=
  continuous_iff_continuousAt.mpr fun a =>
    hf.continuousAt.comp_div_cases _ hg.continuousAt (hh a) (h2h a)


/-- Left multiplication by a nonzero element in a `GroupWithZero` with continuous multiplication
is a homeomorphism of the underlying type. -/
protected def mulLeft₀ (c : α) (hc : c ≠ 0) : α ≃ₜ α :=
  { Equiv.mulLeft₀ c hc with
    continuous_toFun := continuous_mul_left _
    continuous_invFun := continuous_mul_left _ }


/-- Right multiplication by a nonzero element in a `GroupWithZero` with continuous multiplication
is a homeomorphism of the underlying type. -/
protected def mulRight₀ (c : α) (hc : c ≠ 0) : α ≃ₜ α :=
  { Equiv.mulRight₀ c hc with
    continuous_toFun := continuous_mul_right _
    continuous_invFun := continuous_mul_right _ }


@[simp]
theorem coe_mulLeft₀ (c : α) (hc : c ≠ 0) : ⇑(Homeomorph.mulLeft₀ c hc) = (c * ·) :=
  rfl


@[simp]
theorem mulLeft₀_symm_apply (c : α) (hc : c ≠ 0) :
    ((Homeomorph.mulLeft₀ c hc).symm : α → α) = (c⁻¹ * ·) :=
  rfl


@[simp]
theorem coe_mulRight₀ (c : α) (hc : c ≠ 0) : ⇑(Homeomorph.mulRight₀ c hc) = (· * c) :=
  rfl


@[simp]
theorem mulRight₀_symm_apply (c : α) (hc : c ≠ 0) :
    ((Homeomorph.mulRight₀ c hc).symm : α → α) = (· * c⁻¹) :=
  rfl


theorem map_mul_left_nhds₀ (ha : a ≠ 0) (b : G₀) : map (a * ·) (𝓝 b) = 𝓝 (a * b) :=
  (Homeomorph.mulLeft₀ a ha).map_nhds_eq b


theorem map_mul_left_nhds_one₀ (ha : a ≠ 0) : map (a * ·) (𝓝 1) = 𝓝 (a) := by
  /-
    G₀ : Type u_3
    inst✝² : TopologicalSpace G₀
    inst✝¹ : GroupWithZero G₀
    inst✝ : ContinuousMul G₀
    a : G₀
    ha : Ne a 0
    ⊢ Eq (Filter.map (fun x => HMul.hMul a x) (nhds 1)) (nhds a)
  -/
  rw [map_mul_left_nhds₀ ha, mul_one]
  /-
    🎉 no goals
  -/


theorem map_mul_right_nhds₀ (ha : a ≠ 0) (b : G₀) : map (· * a) (𝓝 b) = 𝓝 (b * a) :=
  (Homeomorph.mulRight₀ a ha).map_nhds_eq b


theorem map_mul_right_nhds_one₀ (ha : a ≠ 0) : map (· * a) (𝓝 1) = 𝓝 (a) := by
  /-
    G₀ : Type u_3
    inst✝² : TopologicalSpace G₀
    inst✝¹ : GroupWithZero G₀
    inst✝ : ContinuousMul G₀
    a : G₀
    ha : Ne a 0
    ⊢ Eq (Filter.map (fun x => HMul.hMul x a) (nhds 1)) (nhds a)
  -/
  rw [map_mul_right_nhds₀ ha, one_mul]
  /-
    🎉 no goals
  -/


theorem nhds_translation_mul_inv₀ (ha : a ≠ 0) : comap (· * a⁻¹) (𝓝 1) = 𝓝 a :=
                                                                 /-
                                                                   G₀ : Type u_3
                                                                   inst✝² : TopologicalSpace G₀
                                                                   inst✝¹ : GroupWithZero G₀
                                                                   inst✝ : ContinuousMul G₀
                                                                   a : G₀
                                                                   ha : Ne a 0
                                                                   ⊢ Eq (nhds ((Homeomorph.mulRight₀ a ha).symm.symm 1)) (nhds a)
                                                                 -/
  ((Homeomorph.mulRight₀ a ha).symm.comap_nhds_eq 1).trans <| by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- If a group with zero has continuous multiplication and `fun x ↦ x⁻¹` is continuous at one,
then it is continuous at any unit. -/
theorem HasContinuousInv₀.of_nhds_one (h : Tendsto Inv.inv (𝓝 (1 : G₀)) (𝓝 1)) :
    HasContinuousInv₀ G₀ where
  continuousAt_inv₀ x hx := by
    /-
      G₀ : Type u_3
      inst✝² : TopologicalSpace G₀
      inst✝¹ : GroupWithZero G₀
      inst✝ : ContinuousMul G₀
      h : Filter.Tendsto Inv.inv (nhds 1) (nhds 1)
      x : G₀
      hx : Ne x 0
      ⊢ ContinuousAt Inv.inv x
    -/
    have hx' := inv_ne_zero hx
    rw [ContinuousAt, ← map_mul_left_nhds_one₀ hx, ← nhds_translation_mul_inv₀ hx',
      tendsto_map'_iff, tendsto_comap_iff]
    /-
      G₀ : Type u_3
      inst✝² : TopologicalSpace G₀
      inst✝¹ : GroupWithZero G₀
      inst✝ : ContinuousMul G₀
      h : Filter.Tendsto Inv.inv (nhds 1) (nhds 1)
      x : G₀
      hx : Ne x 0
      hx' : Ne (Inv.inv x) 0
      ⊢ Filter.Tendsto (Function.comp (fun x_1 => HMul.hMul x_1 (Inv.inv (Inv.inv x) …
    -/
    simpa only [Function.comp_def, mul_inv_rev, mul_inv_cancel_right₀ hx']
    /-
      🎉 no goals
    -/


theorem continuousAt_zpow₀ (x : G₀) (m : ℤ) (h : x ≠ 0 ∨ 0 ≤ m) :
    ContinuousAt (fun x => x ^ m) x := by
  /-
    G₀ : Type u_3
    inst✝³ : GroupWithZero G₀
    inst✝² : TopologicalSpace G₀
    inst✝¹ : HasContinuousInv₀ G₀
    inst✝ : ContinuousMul G₀
    x : G₀
    m : Int
    h : Or (Ne x 0) (LE.le 0 m)
    ⊢ ContinuousAt (fun x => HPow.hPow x m) x
  -/
  cases' m with m m
    /-
      case ofNat
      G₀ : Type u_3
      inst✝³ : GroupWithZero G₀
      inst✝² : TopologicalSpace G₀
      inst✝¹ : HasContinuousInv₀ G₀
      inst✝ : ContinuousMul G₀
      x : G₀
      m : Nat
      h : Or (Ne x 0) (LE.le 0 (Int.ofNat m))
      ⊢ ContinuousAt (fun x => HPow.hPow x (Int.ofNat m)) x
    -/
  · simpa only [Int.ofNat_eq_coe, zpow_natCast] using continuousAt_pow x m
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      G₀ : Type u_3
      inst✝³ : GroupWithZero G₀
      inst✝² : TopologicalSpace G₀
      inst✝¹ : HasContinuousInv₀ G₀
      inst✝ : ContinuousMul G₀
      x : G₀
      m : Nat
      h : Or (Ne x 0) (LE.le 0 (Int.negSucc m))
      ⊢ ContinuousAt (fun x => HPow.hPow x (Int.negSucc m)) x
    -/
  · simp only [zpow_negSucc]
    /-
      case negSucc
      G₀ : Type u_3
      inst✝³ : GroupWithZero G₀
      inst✝² : TopologicalSpace G₀
      inst✝¹ : HasContinuousInv₀ G₀
      inst✝ : ContinuousMul G₀
      x : G₀
      m : Nat
      h : Or (Ne x 0) (LE.le 0 (Int.negSucc m))
      ⊢ ContinuousAt (fun x => Inv.inv (HPow.hPow x (HAdd.hAdd m 1))) x
    -/
    have hx : x ≠ 0 := h.resolve_right (Int.negSucc_lt_zero m).not_le
    /-
      case negSucc
      G₀ : Type u_3
      inst✝³ : GroupWithZero G₀
      inst✝² : TopologicalSpace G₀
      inst✝¹ : HasContinuousInv₀ G₀
      inst✝ : ContinuousMul G₀
      x : G₀
      m : Nat
      h : Or (Ne x 0) (LE.le 0 (Int.negSucc m))
      hx : Ne x 0
      ⊢ ContinuousAt (fun x => Inv.inv (HPow.hPow x (HAdd.hAdd m 1))) x
    -/
    exact (continuousAt_pow x (m + 1)).inv₀ (pow_ne_zero _ hx)
    /-
      🎉 no goals
    -/


theorem continuousOn_zpow₀ (m : ℤ) : ContinuousOn (fun x : G₀ => x ^ m) {0}ᶜ := fun _x hx =>
  (continuousAt_zpow₀ _ _ (Or.inl hx)).continuousWithinAt


theorem Filter.Tendsto.zpow₀ {f : α → G₀} {l : Filter α} {a : G₀} (hf : Tendsto f l (𝓝 a)) (m : ℤ)
    (h : a ≠ 0 ∨ 0 ≤ m) : Tendsto (fun x => f x ^ m) l (𝓝 (a ^ m)) :=
  (continuousAt_zpow₀ _ m h).tendsto.comp hf


@[fun_prop]
nonrec theorem ContinuousAt.zpow₀ (hf : ContinuousAt f a) (m : ℤ) (h : f a ≠ 0 ∨ 0 ≤ m) :
    ContinuousAt (fun x => f x ^ m) a :=
  hf.zpow₀ m h


nonrec theorem ContinuousWithinAt.zpow₀ (hf : ContinuousWithinAt f s a) (m : ℤ)
    (h : f a ≠ 0 ∨ 0 ≤ m) : ContinuousWithinAt (fun x => f x ^ m) s a :=
  hf.zpow₀ m h


@[fun_prop]
theorem ContinuousOn.zpow₀ (hf : ContinuousOn f s) (m : ℤ) (h : ∀ a ∈ s, f a ≠ 0 ∨ 0 ≤ m) :
    ContinuousOn (fun x => f x ^ m) s := fun a ha => (hf a ha).zpow₀ m (h a ha)


@[continuity, fun_prop]
theorem Continuous.zpow₀ (hf : Continuous f) (m : ℤ) (h0 : ∀ a, f a ≠ 0 ∨ 0 ≤ m) :
    Continuous fun x => f x ^ m :=
  continuous_iff_continuousAt.2 fun x => (hf.tendsto x).zpow₀ m (h0 x)


