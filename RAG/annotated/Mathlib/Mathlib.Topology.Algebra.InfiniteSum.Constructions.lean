@[to_additive]
theorem hasProd_pi_single [DecidableEq β] (b : β) (a : α) : HasProd (Pi.mulSingle b a) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : DecidableEq β
    b : β
    a : α
    ⊢ HasProd (Pi.mulSingle b a) a
  -/
  convert hasProd_ite_eq b a
  /-
    case h.e'_5.h
    α : Type u_1
    β : Type u_2
    inst✝² : CommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : DecidableEq β
    b : β
    a : α
    x✝ : β
    ⊢ Eq (Pi.mulSingle b a x✝) (ite (Eq x✝ b) a 1)
  -/
  simp [Pi.mulSingle_apply]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem tprod_pi_single [DecidableEq β] (b : β) (a : α) : ∏' b', Pi.mulSingle b a b' = a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : DecidableEq β
    b : β
    a : α
    ⊢ Eq (tprod fun b' => Pi.mulSingle b a b') a
  -/
  rw [tprod_eq_mulSingle b]
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : CommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : DecidableEq β
      b : β
      a : α
      ⊢ Eq (Pi.mulSingle b a b) a
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : CommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : DecidableEq β
      b : β
      a : α
      ⊢ ∀ (b' : β), Ne b' b → Eq (Pi.mulSingle b a b') 1
    -/
  · intro b' hb'; simp [hb']
                  /-
                    🎉 no goals
                  -/


@[to_additive tsum_setProd_singleton_left]
lemma tprod_setProd_singleton_left (b : β) (t : Set γ) (f : β × γ → α) :
    (∏' x : {b} ×ˢ t, f x) = ∏' c : t, f (b, c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    b : β
    t : Set γ
    f : Prod β γ → α
    ⊢ Eq (tprod fun x => f ↑x) (tprod fun c => f { fst := b, snd := ↑c })
  -/
  rw [tprod_congr_set_coe _ Set.singleton_prod, tprod_image _ (Prod.mk.inj_left b).injOn]
  /-
    🎉 no goals
  -/


@[to_additive tsum_setProd_singleton_right]
lemma tprod_setProd_singleton_right (s : Set β) (c : γ) (f : β × γ → α) :
    (∏' x : s ×ˢ {c}, f x) = ∏' b : s, f (b, c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    s : Set β
    c : γ
    f : Prod β γ → α
    ⊢ Eq (tprod fun x => f ↑x) (tprod fun b => f { fst := ↑b, snd := c })
  -/
  rw [tprod_congr_set_coe _ Set.prod_singleton, tprod_image _ (Prod.mk.inj_right c).injOn]
  /-
    🎉 no goals
  -/


@[to_additive Summable.prod_symm]
theorem Multipliable.prod_symm {f : β × γ → α} (hf : Multipliable f) :
    Multipliable fun p : γ × β ↦ f p.swap :=
  (Equiv.prodComm γ β).multipliable_iff.2 hf


@[to_additive HasSum.prod_mk]
theorem HasProd.prod_mk {f : β → α} {g : β → γ} {a : α} {b : γ}
    (hf : HasProd f a) (hg : HasProd g b) : HasProd (fun x ↦ (⟨f x, g x⟩ : α × γ)) ⟨a, b⟩ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : CommMonoid α
    inst✝² : TopologicalSpace α
    inst✝¹ : CommMonoid γ
    inst✝ : TopologicalSpace γ
    f : β → α
    g : β → γ
    a : α
    b : γ
    hf : HasProd f a
    hg : HasProd g b
    ⊢ HasProd (fun x => { fst := f x, snd := g x }) { fst := a, snd := b }
  -/
  simp [HasProd, ← prod_mk_prod, Filter.Tendsto.prod_mk_nhds hf hg]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem HasProd.sigma {γ : β → Type*} {f : (Σ b : β, γ b) → α} {g : β → α} {a : α}
    (ha : HasProd f a) (hf : ∀ b, HasProd (fun c ↦ f ⟨b, c⟩) (g b)) : HasProd g a := by
  classical
  refine (atTop_basis.tendsto_iff (closed_nhds_basis a)).mpr ?_
  rintro s ⟨hs, hsc⟩
  rcases mem_atTop_sets.mp (ha hs) with ⟨u, hu⟩
  use u.image Sigma.fst, trivial
  intro bs hbs
  simp only [Set.mem_preimage, Finset.le_iff_subset] at hu
  have : Tendsto (fun t : Finset (Σb, γ b) ↦ ∏ p ∈ t with p.1 ∈ bs, f p) atTop
      (𝓝 <| ∏ b ∈ bs, g b) := by
    simp only [← sigma_preimage_mk, prod_sigma]
    refine tendsto_finset_prod _ fun b _ ↦ ?_
    change
      Tendsto (fun t ↦ (fun t ↦ ∏ s ∈ t, f ⟨b, s⟩) (preimage t (Sigma.mk b) _)) atTop (𝓝 (g b))
    exact (hf b).comp (tendsto_finset_preimage_atTop_atTop (sigma_mk_injective))
  refine hsc.mem_of_tendsto this (eventually_atTop.2 ⟨u, fun t ht ↦ hu _ fun x hx ↦ ?_⟩)
  exact mem_filter.2 ⟨ht hx, hbs <| mem_image_of_mem _ hx⟩


/-- If a function `f` on `β × γ` has product `a` and for each `b` the restriction of `f` to
`{b} × γ` has product `g b`, then the function `g` has product `a`. -/
@[to_additive HasSum.prod_fiberwise "If a series `f` on `β × γ` has sum `a` and for each `b` the
restriction of `f` to `{b} × γ` has sum `g b`, then the series `g` has sum `a`."]
theorem HasProd.prod_fiberwise {f : β × γ → α} {g : β → α} {a : α} (ha : HasProd f a)
    (hf : ∀ b, HasProd (fun c ↦ f (b, c)) (g b)) : HasProd g a :=
  HasProd.sigma ((Equiv.sigmaEquivProd β γ).hasProd_iff.2 ha) hf


@[to_additive]
theorem Multipliable.sigma' {γ : β → Type*} {f : (Σb : β, γ b) → α} (ha : Multipliable f)
    (hf : ∀ b, Multipliable fun c ↦ f ⟨b, c⟩) : Multipliable fun b ↦ ∏' c, f ⟨b, c⟩ :=
  (ha.hasProd.sigma fun b ↦ (hf b).hasProd).multipliable


@[to_additive]
theorem HasProd.sigma_of_hasProd {γ : β → Type*} {f : (Σb : β, γ b) → α} {g : β → α}
    {a : α} (ha : HasProd g a) (hf : ∀ b, HasProd (fun c ↦ f ⟨b, c⟩) (g b)) (hf' : Multipliable f) :
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝³ : CommMonoid α
                        inst✝² : TopologicalSpace α
                        inst✝¹ : ContinuousMul α
                        inst✝ : T3Space α
                        γ : β → Type u_4
                        f : (Sigma fun b => γ b) → α
                        g : β → α
                        a : α
                        ha : HasProd g a
                        hf : ∀ (b : β), HasProd (fun c => f ⟨b, c⟩) (g b)
                        hf' : Multipliable f
                        ⊢ HasProd f a
                      -/
    HasProd f a := by simpa [(hf'.hasProd.sigma hf).unique ha] using hf'.hasProd
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem tprod_sigma' {γ : β → Type*} {f : (Σb : β, γ b) → α}
    (h₁ : ∀ b, Multipliable fun c ↦ f ⟨b, c⟩) (h₂ : Multipliable f) :
    ∏' p, f p = ∏' (b) (c), f ⟨b, c⟩ :=
  (h₂.hasProd.sigma fun b ↦ (h₁ b).hasProd).tprod_eq.symm


@[to_additive tsum_prod']
theorem tprod_prod' {f : β × γ → α} (h : Multipliable f)
    (h₁ : ∀ b, Multipliable fun c ↦ f (b, c)) :
    ∏' p, f p = ∏' (b) (c), f (b, c) :=
  (h.hasProd.prod_fiberwise fun b ↦ (h₁ b).hasProd).tprod_eq.symm


@[to_additive]
theorem tprod_comm' {f : β → γ → α} (h : Multipliable (Function.uncurry f))
    (h₁ : ∀ b, Multipliable (f b)) (h₂ : ∀ c, Multipliable fun b ↦ f b c) :
    ∏' (c) (b), f b c = ∏' (b) (c), f b c := by
  erw [← tprod_prod' h h₁, ← tprod_prod' h.prod_symm h₂,
      ← (Equiv.prodComm γ β).tprod_eq (uncurry f)]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : CommMonoid α
    inst✝² : TopologicalSpace α
    inst✝¹ : ContinuousMul α
    inst✝ : T3Space α
    f : β → γ → α
    h : Multipliable (Function.uncurry f)
    h₁ : ∀ (b : β), Multipliable (f b)
    h₂ : ∀ (c : γ), Multipliable fun b => f b c
    ⊢ Eq (tprod fun p => Function.uncurry f p.swap) (tprod fun c => Function.uncur …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem HasProd.of_sigma {γ : β → Type*} {f : (Σ b : β, γ b) → α} {g : β → α} {a : α}
    (hf : ∀ b, HasProd (fun c ↦ f ⟨b, c⟩) (g b)) (hg : HasProd g a)
    (h : CauchySeq (fun (s : Finset (Σ b : β, γ b)) ↦ ∏ i ∈ s, f i)) :
    HasProd f a := by
  classical
  apply le_nhds_of_cauchy_adhp h
  simp only [← mapClusterPt_def, mapClusterPt_iff, frequently_atTop, ge_iff_le, le_eq_subset]
  intro u hu s
  rcases mem_nhds_iff.1 hu with ⟨v, vu, v_open, hv⟩
  obtain ⟨t0, st0, ht0⟩ : ∃ t0, ∏ i ∈ t0, g i ∈ v ∧ s.image Sigma.fst ⊆ t0 := by
    have A : ∀ᶠ t0 in (atTop : Filter (Finset β)), ∏ i ∈ t0, g i ∈ v := hg (v_open.mem_nhds hv)
    exact (A.and (Ici_mem_atTop _)).exists
  have L : Tendsto (fun t : Finset (Σb, γ b) ↦ ∏ p ∈ t with p.1 ∈ t0, f p) atTop
      (𝓝 <| ∏ b ∈ t0, g b) := by
    simp only [← sigma_preimage_mk, prod_sigma]
    refine tendsto_finset_prod _ fun b _ ↦ ?_
    change
      Tendsto (fun t ↦ (fun t ↦ ∏ s ∈ t, f ⟨b, s⟩) (preimage t (Sigma.mk b) _)) atTop (𝓝 (g b))
    exact (hf b).comp (tendsto_finset_preimage_atTop_atTop (sigma_mk_injective))
  have : ∃ t, ∏ p ∈ t with p.1 ∈ t0, f p ∈ v ∧ s ⊆ t :=
    ((Tendsto.eventually_mem L (v_open.mem_nhds st0)).and (Ici_mem_atTop _)).exists
  obtain ⟨t, tv, st⟩ := this
  refine ⟨{p ∈ t | p.1 ∈ t0}, fun x hx ↦ ?_, vu tv⟩
  simpa only [mem_filter, st hx, true_and] using ht0 (mem_image_of_mem Sigma.fst hx)


@[to_additive]
theorem Multipliable.sigma_factor {γ : β → Type*} {f : (Σb : β, γ b) → α}
    (ha : Multipliable f) (b : β) :
    Multipliable fun c ↦ f ⟨b, c⟩ :=
  ha.comp_injective sigma_mk_injective


@[to_additive]
theorem Multipliable.sigma {γ : β → Type*} {f : (Σb : β, γ b) → α} (ha : Multipliable f) :
    Multipliable fun b ↦ ∏' c, f ⟨b, c⟩ :=
  ha.sigma' fun b ↦ ha.sigma_factor b


@[to_additive Summable.prod_factor]
theorem Multipliable.prod_factor {f : β × γ → α} (h : Multipliable f) (b : β) :
    Multipliable fun c ↦ f (b, c) :=
  h.comp_injective fun _ _ h ↦ (Prod.ext_iff.1 h).2


@[to_additive Summable.prod]
lemma Multipliable.prod {f : β × γ → α} (h : Multipliable f) :
    Multipliable fun b ↦ ∏' c, f (b, c) :=
  ((Equiv.sigmaEquivProd β γ).multipliable_iff.mpr h).sigma


@[to_additive]
lemma HasProd.tprod_fiberwise [T2Space α] {f : β → α} {a : α} (hf : HasProd f a) (g : β → γ) :
    HasProd (fun c : γ ↦ ∏' b : g ⁻¹' {c}, f b) a :=
  (((Equiv.sigmaFiberEquiv g).hasProd_iff).mpr hf).sigma <|
    fun _ ↦ ((hf.multipliable.subtype _).hasProd_iff).mpr rfl


@[to_additive]
theorem tprod_sigma {γ : β → Type*} {f : (Σb : β, γ b) → α} (ha : Multipliable f) :
    ∏' p, f p = ∏' (b) (c), f ⟨b, c⟩ :=
  tprod_sigma' (fun b ↦ ha.sigma_factor b) ha


@[to_additive tsum_prod]
theorem tprod_prod {f : β × γ → α} (h : Multipliable f) :
    ∏' p, f p = ∏' (b) (c), f ⟨b, c⟩ :=
  tprod_prod' h h.prod_factor


@[to_additive]
theorem tprod_comm {f : β → γ → α} (h : Multipliable (Function.uncurry f)) :
    ∏' (c) (b), f b c = ∏' (b) (c), f b c :=
  tprod_comm' h h.prod_factor h.prod_symm.prod_factor


@[to_additive]
theorem Pi.hasProd {f : ι → ∀ x, π x} {g : ∀ x, π x} :
    HasProd f g ↔ ∀ x, HasProd (fun i ↦ f i x) (g x) := by
  /-
    α : Type u_1
    ι : Type u_4
    π : α → Type u_5
    inst✝¹ : (x : α) → CommMonoid (π x)
    inst✝ : (x : α) → TopologicalSpace (π x)
    f : ι → (x : α) → π x
    g : (x : α) → π x
    ⊢ Iff (HasProd f g) (∀ (x : α), HasProd (fun i => f i x) (g x))
  -/
  simp only [HasProd, tendsto_pi_nhds, prod_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Pi.multipliable {f : ι → ∀ x, π x} : Multipliable f ↔ ∀ x, Multipliable fun i ↦ f i x := by
  /-
    α : Type u_1
    ι : Type u_4
    π : α → Type u_5
    inst✝¹ : (x : α) → CommMonoid (π x)
    inst✝ : (x : α) → TopologicalSpace (π x)
    f : ι → (x : α) → π x
    ⊢ Iff (Multipliable f) (∀ (x : α), Multipliable fun i => f i x)
  -/
  simp only [Multipliable, Pi.hasProd, Classical.skolem]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tprod_apply [∀ x, T2Space (π x)] {f : ι → ∀ x, π x} {x : α} (hf : Multipliable f) :
    (∏' i, f i) x = ∏' i, f i x :=
  (Pi.hasProd.mp hf.hasProd x).tprod_eq.symm


theorem HasSum.op (hf : HasSum f a) : HasSum (fun a ↦ op (f a)) (op a) :=
  (hf.map (@opAddEquiv α _) continuous_op : _)


theorem Summable.op (hf : Summable f) : Summable (op ∘ f) :=
  hf.hasSum.op.summable


theorem HasSum.unop {f : β → αᵐᵒᵖ} {a : αᵐᵒᵖ} (hf : HasSum f a) :
    HasSum (fun a ↦ unop (f a)) (unop a) :=
  (hf.map (@opAddEquiv α _).symm continuous_unop : _)


theorem Summable.unop {f : β → αᵐᵒᵖ} (hf : Summable f) : Summable (unop ∘ f) :=
  hf.hasSum.unop.summable


@[simp]
theorem hasSum_op : HasSum (fun a ↦ op (f a)) (op a) ↔ HasSum f a :=
  ⟨HasSum.unop, HasSum.op⟩


@[simp]
theorem hasSum_unop {f : β → αᵐᵒᵖ} {a : αᵐᵒᵖ} :
    HasSum (fun a ↦ unop (f a)) (unop a) ↔ HasSum f a :=
  ⟨HasSum.op, HasSum.unop⟩


@[simp]
theorem summable_op : (Summable fun a ↦ op (f a)) ↔ Summable f :=
  ⟨Summable.unop, Summable.op⟩

-- Porting note: This theorem causes a loop easily in Lean 4, so the priority should be `low`.

@[simp low]
theorem summable_unop {f : β → αᵐᵒᵖ} : (Summable fun a ↦ unop (f a)) ↔ Summable f :=
  ⟨Summable.op, Summable.unop⟩


theorem tsum_op [T2Space α] :
    ∑' x, op (f x) = op (∑' x, f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : AddCommMonoid α
    inst✝¹ : TopologicalSpace α
    f : β → α
    inst✝ : T2Space α
    ⊢ Eq (tsum fun x => MulOpposite.op (f x)) (MulOpposite.op (tsum fun x => f x))
  -/
  by_cases h : Summable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : AddCommMonoid α
      inst✝¹ : TopologicalSpace α
      f : β → α
      inst✝ : T2Space α
      h : Summable f
      ⊢ Eq (tsum fun x => MulOpposite.op (f x)) (MulOpposite.op (tsum fun x => f x))
    -/
  · exact h.hasSum.op.tsum_eq
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : AddCommMonoid α
      inst✝¹ : TopologicalSpace α
      f : β → α
      inst✝ : T2Space α
      h : Not (Summable f)
      ⊢ Eq (tsum fun x => MulOpposite.op (f x)) (MulOpposite.op (tsum fun x => f x))
    -/
  · have ho := summable_op.not.mpr h
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : AddCommMonoid α
      inst✝¹ : TopologicalSpace α
      f : β → α
      inst✝ : T2Space α
      h : Not (Summable f)
      ho : Not (Summable fun a => MulOpposite.op (f a))
      ⊢ Eq (tsum fun x => MulOpposite.op (f x)) (MulOpposite.op (tsum fun x => f x))
    -/
    rw [tsum_eq_zero_of_not_summable h, tsum_eq_zero_of_not_summable ho, op_zero]
    /-
      🎉 no goals
    -/


theorem tsum_unop [T2Space α] {f : β → αᵐᵒᵖ} :
    ∑' x, unop (f x) = unop (∑' x, f x) :=
  op_injective tsum_op.symm


theorem HasSum.star (h : HasSum f a) : HasSum (fun b ↦ star (f b)) (star a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : AddCommMonoid α
    inst✝² : TopologicalSpace α
    inst✝¹ : StarAddMonoid α
    inst✝ : ContinuousStar α
    f : β → α
    a : α
    h : HasSum f a
    ⊢ HasSum (fun b => Star.star (f b)) (Star.star a)
  -/
  simpa only using h.map (starAddEquiv : α ≃+ α) continuous_star
  /-
    🎉 no goals
  -/


theorem Summable.star (hf : Summable f) : Summable fun b ↦ star (f b) :=
  hf.hasSum.star.summable


theorem Summable.ofStar (hf : Summable fun b ↦ Star.star (f b)) : Summable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : AddCommMonoid α
    inst✝² : TopologicalSpace α
    inst✝¹ : StarAddMonoid α
    inst✝ : ContinuousStar α
    f : β → α
    hf : Summable fun b => Star.star (f b)
    ⊢ Summable f
  -/
  simpa only [star_star] using hf.star
  /-
    🎉 no goals
  -/


@[simp]
theorem summable_star_iff : (Summable fun b ↦ star (f b)) ↔ Summable f :=
  ⟨Summable.ofStar, Summable.star⟩


@[simp]
theorem summable_star_iff' : Summable (star f) ↔ Summable f :=
  summable_star_iff


theorem tsum_star [T2Space α] : star (∑' b, f b) = ∑' b, star (f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : AddCommMonoid α
    inst✝³ : TopologicalSpace α
    inst✝² : StarAddMonoid α
    inst✝¹ : ContinuousStar α
    f : β → α
    inst✝ : T2Space α
    ⊢ Eq (Star.star (tsum fun b => f b)) (tsum fun b => Star.star (f b))
  -/
  by_cases hf : Summable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝⁴ : AddCommMonoid α
      inst✝³ : TopologicalSpace α
      inst✝² : StarAddMonoid α
      inst✝¹ : ContinuousStar α
      f : β → α
      inst✝ : T2Space α
      hf : Summable f
      ⊢ Eq (Star.star (tsum fun b => f b)) (tsum fun b => Star.star (f b))
    -/
  · exact hf.hasSum.star.tsum_eq.symm
    /-
      🎉 no goals
    -/
  · rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable (mt Summable.ofStar hf),
      star_zero]


