@[to_additive]
theorem HasProd.inv (h : HasProd f a) : HasProd (fun b ↦ (f b)⁻¹) a⁻¹ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f : β → α
    a : α
    h : HasProd f a
    ⊢ HasProd (fun b => Inv.inv (f b)) (Inv.inv a)
  -/
  simpa only using h.map (MonoidHom.id α)⁻¹ continuous_inv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multipliable.inv (hf : Multipliable f) : Multipliable fun b ↦ (f b)⁻¹ :=
  hf.hasProd.inv.multipliable


@[to_additive]
theorem Multipliable.of_inv (hf : Multipliable fun b ↦ (f b)⁻¹) : Multipliable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f : β → α
    hf : Multipliable fun b => Inv.inv (f b)
    ⊢ Multipliable f
  -/
  simpa only [inv_inv] using hf.inv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multipliable_inv_iff : (Multipliable fun b ↦ (f b)⁻¹) ↔ Multipliable f :=
  ⟨Multipliable.of_inv, Multipliable.inv⟩


@[to_additive]
theorem HasProd.div (hf : HasProd f a₁) (hg : HasProd g a₂) :
    HasProd (fun b ↦ f b / g b) (a₁ / a₂) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f g : β → α
    a₁ a₂ : α
    hf : HasProd f a₁
    hg : HasProd g a₂
    ⊢ HasProd (fun b => HDiv.hDiv (f b) (g b)) (HDiv.hDiv a₁ a₂)
  -/
  simp only [div_eq_mul_inv]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f g : β → α
    a₁ a₂ : α
    hf : HasProd f a₁
    hg : HasProd g a₂
    ⊢ HasProd (fun b => HMul.hMul (f b) (Inv.inv (g b))) (HMul.hMul a₁ (Inv.inv a₂))
  -/
  exact hf.mul hg.inv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multipliable.div (hf : Multipliable f) (hg : Multipliable g) :
    Multipliable fun b ↦ f b / g b :=
  (hf.hasProd.div hg.hasProd).multipliable


@[to_additive]
theorem Multipliable.trans_div (hg : Multipliable g) (hfg : Multipliable fun b ↦ f b / g b) :
    Multipliable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f g : β → α
    hg : Multipliable g
    hfg : Multipliable fun b => HDiv.hDiv (f b) (g b)
    ⊢ Multipliable f
  -/
  simpa only [div_mul_cancel] using hfg.mul hg
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multipliable_iff_of_multipliable_div (hfg : Multipliable fun b ↦ f b / g b) :
    Multipliable f ↔ Multipliable g :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 inst✝² : CommGroup α
                                 inst✝¹ : TopologicalSpace α
                                 inst✝ : TopologicalGroup α
                                 f g : β → α
                                 hfg : Multipliable fun b => HDiv.hDiv (f b) (g b)
                                 hf : Multipliable f
                                 ⊢ Multipliable fun b => HDiv.hDiv (g b) (f b)
                               -/
  ⟨fun hf ↦ hf.trans_div <| by simpa only [inv_div] using hfg.inv, fun hg ↦ hg.trans_div hfg⟩
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
theorem HasProd.update (hf : HasProd f a₁) (b : β) [DecidableEq β] (a : α) :
    HasProd (update f b a) (a / f b * a₁) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : CommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalGroup α
    f : β → α
    a₁ : α
    hf : HasProd f a₁
    b : β
    inst✝ : DecidableEq β
    a : α
    ⊢ HasProd (Function.update f b a) (HMul.hMul (HDiv.hDiv a (f b)) a₁)
  -/
  convert (hasProd_ite_eq b (a / f b)).mul hf with b'
  /-
    case h.e'_5.h
    α : Type u_1
    β : Type u_2
    inst✝³ : CommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalGroup α
    f : β → α
    a₁ : α
    hf : HasProd f a₁
    b : β
    inst✝ : DecidableEq β
    a : α
    b' : β
    ⊢ Eq (Function.update f b a b') (HMul.hMul (ite (Eq b' b) (HDiv.hDiv a (f b))  …
  -/
  by_cases h : b' = b
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      a₁ : α
      hf : HasProd f a₁
      b : β
      inst✝ : DecidableEq β
      a : α
      b' : β
      h : Eq b' b
      ⊢ Eq (Function.update f b a b') (HMul.hMul (ite (Eq b' b) (HDiv.hDiv a (f b))  …
    -/
  · rw [h, update_self]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      a₁ : α
      hf : HasProd f a₁
      b : β
      inst✝ : DecidableEq β
      a : α
      b' : β
      h : Eq b' b
      ⊢ Eq a (HMul.hMul (ite (Eq b b) (HDiv.hDiv a (f b)) 1) (f b))
    -/
    simp [eq_self_iff_true, if_true, sub_add_cancel]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      a₁ : α
      hf : HasProd f a₁
      b : β
      inst✝ : DecidableEq β
      a : α
      b' : β
      h : Not (Eq b' b)
      ⊢ Eq (Function.update f b a b') (HMul.hMul (ite (Eq b' b) (HDiv.hDiv a (f b))  …
    -/
  · simp only [h, update_of_ne, if_false, Ne, one_mul, not_false_iff]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem Multipliable.update (hf : Multipliable f) (b : β) [DecidableEq β] (a : α) :
    Multipliable (update f b a) :=
  (hf.hasProd.update b a).multipliable


@[to_additive]
theorem HasProd.hasProd_compl_iff {s : Set β} (hf : HasProd (f ∘ (↑) : s → α) a₁) :
    HasProd (f ∘ (↑) : ↑sᶜ → α) a₂ ↔ HasProd f (a₁ * a₂) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f : β → α
    a₁ a₂ : α
    s : Set β
    hf : HasProd (Function.comp f Subtype.val) a₁
    ⊢ Iff (HasProd (Function.comp f Subtype.val) a₂) (HasProd f (HMul.hMul a₁ a₂))
  -/
  refine ⟨fun h ↦ hf.mul_compl h, fun h ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f : β → α
    a₁ a₂ : α
    s : Set β
    hf : HasProd (Function.comp f Subtype.val) a₁
    h : HasProd f (HMul.hMul a₁ a₂)
    ⊢ HasProd (Function.comp f Subtype.val) a₂
  -/
  rw [hasProd_subtype_iff_mulIndicator] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f : β → α
    a₁ a₂ : α
    s : Set β
    hf : HasProd (s.mulIndicator f) a₁
    h : HasProd f (HMul.hMul a₁ a₂)
    ⊢ HasProd ((HasCompl.compl s).mulIndicator f) a₂
  -/
  rw [Set.mulIndicator_compl]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalGroup α
    f : β → α
    a₁ a₂ : α
    s : Set β
    hf : HasProd (s.mulIndicator f) a₁
    h : HasProd f (HMul.hMul a₁ a₂)
    ⊢ HasProd (HMul.hMul f (Inv.inv (s.mulIndicator f))) a₂
  -/
  simpa only [div_eq_mul_inv, mul_inv_cancel_comm] using h.div hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem HasProd.hasProd_iff_compl {s : Set β} (hf : HasProd (f ∘ (↑) : s → α) a₁) :
    HasProd f a₂ ↔ HasProd (f ∘ (↑) : ↑sᶜ → α) (a₂ / a₁) :=
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 inst✝² : CommGroup α
                                                 inst✝¹ : TopologicalSpace α
                                                 inst✝ : TopologicalGroup α
                                                 f : β → α
                                                 a₁ a₂ : α
                                                 s : Set β
                                                 hf : HasProd (Function.comp f Subtype.val) a₁
                                                 ⊢ Iff (HasProd f (HMul.hMul a₁ (HDiv.hDiv a₂ a₁))) (HasProd f a₂)
                                               -/
  Iff.symm <| hf.hasProd_compl_iff.trans <| by rw [mul_div_cancel]
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem Multipliable.multipliable_compl_iff {s : Set β} (hf : Multipliable (f ∘ (↑) : s → α)) :
    Multipliable (f ∘ (↑) : ↑sᶜ → α) ↔ Multipliable f where
  mp := fun ⟨_, ha⟩ ↦ (hf.hasProd.hasProd_compl_iff.1 ha).multipliable
  mpr := fun ⟨_, ha⟩ ↦ (hf.hasProd.hasProd_iff_compl.1 ha).multipliable


@[to_additive]
protected theorem Finset.hasProd_compl_iff (s : Finset β) :
    HasProd (fun x : { x // x ∉ s } ↦ f x) a ↔ HasProd f (a * ∏ i ∈ s, f i) :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                inst✝² : CommGroup α
                                                inst✝¹ : TopologicalSpace α
                                                inst✝ : TopologicalGroup α
                                                f : β → α
                                                a : α
                                                s : Finset β
                                                ⊢ Iff (HasProd f (HMul.hMul (s.prod fun b => f b) a)) (HasProd f (HMul.hMul a  …
                                              -/
  (s.hasProd f).hasProd_compl_iff.trans <| by rw [mul_comm]
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
protected theorem Finset.hasProd_iff_compl (s : Finset β) :
    HasProd f a ↔ HasProd (fun x : { x // x ∉ s } ↦ f x) (a / ∏ i ∈ s, f i) :=
  (s.hasProd f).hasProd_iff_compl


@[to_additive]
protected theorem Finset.multipliable_compl_iff (s : Finset β) :
    (Multipliable fun x : { x // x ∉ s } ↦ f x) ↔ Multipliable f :=
  (s.multipliable f).multipliable_compl_iff


@[to_additive]
theorem Set.Finite.multipliable_compl_iff {s : Set β} (hs : s.Finite) :
    Multipliable (f ∘ (↑) : ↑sᶜ → α) ↔ Multipliable f :=
  (hs.multipliable f).multipliable_compl_iff


@[to_additive]
theorem hasProd_ite_div_hasProd [DecidableEq β] (hf : HasProd f a) (b : β) :
    HasProd (fun n ↦ ite (n = b) 1 (f n)) (a / f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : CommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalGroup α
    f : β → α
    a : α
    inst✝ : DecidableEq β
    hf : HasProd f a
    b : β
    ⊢ HasProd (fun n => ite (Eq n b) 1 (f n)) (HDiv.hDiv a (f b))
  -/
  convert hf.update b 1 using 1
    /-
      case h.e'_5
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      a : α
      inst✝ : DecidableEq β
      hf : HasProd f a
      b : β
      ⊢ Eq (fun n => ite (Eq n b) 1 (f n)) (Function.update f b 1)
    -/
  · ext n
    /-
      case h.e'_5.h
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      a : α
      inst✝ : DecidableEq β
      hf : HasProd f a
      b n : β
      ⊢ Eq (ite (Eq n b) 1 (f n)) (Function.update f b 1 n)
    -/
    rw [Function.update_apply]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      a : α
      inst✝ : DecidableEq β
      hf : HasProd f a
      b : β
      ⊢ Eq (HDiv.hDiv a (f b)) (HMul.hMul (HDiv.hDiv 1 (f b)) a)
    -/
  · rw [div_mul_eq_mul_div, one_mul]
    /-
      🎉 no goals
    -/


/-- A more general version of `Multipliable.congr`, allowing the functions to
disagree on a finite set. -/
@[to_additive "A more general version of `Summable.congr`, allowing the functions to
disagree on a finite set."]
theorem Multipliable.congr_cofinite (hf : Multipliable f) (hfg : f =ᶠ[cofinite] g) :
    Multipliable g :=
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   β : Type u_2
                                                                                   inst✝² : CommGroup α
                                                                                   inst✝¹ : TopologicalSpace α
                                                                                   inst✝ : TopologicalGroup α
                                                                                   f g : β → α
                                                                                   hf : Multipliable f
                                                                                   hfg : Filter.cofinite.EventuallyEq f g
                                                                                   ⊢ ∀ (b : ↑(HasCompl.compl (HasCompl.compl (setOf fun x => (fun x => Eq (f x) ( …
                                                                                 -/
  hfg.multipliable_compl_iff.mp <| (hfg.multipliable_compl_iff.mpr hf).congr (by simp)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- A more general version of `multipliable_congr`, allowing the functions to
disagree on a finite set. -/
@[to_additive "A more general version of `summable_congr`, allowing the functions to
disagree on a finite set."]
theorem multipliable_congr_cofinite (hfg : f =ᶠ[cofinite] g) :
    Multipliable f ↔ Multipliable g :=
  ⟨fun h ↦ h.congr_cofinite hfg, fun h ↦ h.congr_cofinite (hfg.mono fun _ h' ↦ h'.symm)⟩


@[to_additive]
theorem Multipliable.congr_atTop {f₁ g₁ : ℕ → α} (hf : Multipliable f₁) (hfg : f₁ =ᶠ[atTop] g₁) :
    Multipliable g₁ := hf.congr_cofinite (Nat.cofinite_eq_atTop ▸ hfg)


@[to_additive]
theorem multipliable_congr_atTop {f₁ g₁ : ℕ → α} (hfg : f₁ =ᶠ[atTop] g₁) :
    Multipliable f₁ ↔ Multipliable g₁ := multipliable_congr_cofinite (Nat.cofinite_eq_atTop ▸ hfg)


@[to_additive]
theorem tprod_inv : ∏' b, (f b)⁻¹ = (∏' b, f b)⁻¹ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : CommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalGroup α
    f : β → α
    inst✝ : T2Space α
    ⊢ Eq (tprod fun b => Inv.inv (f b)) (Inv.inv (tprod fun b => f b))
  -/
  by_cases hf : Multipliable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝³ : CommGroup α
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalGroup α
      f : β → α
      inst✝ : T2Space α
      hf : Multipliable f
      ⊢ Eq (tprod fun b => Inv.inv (f b)) (Inv.inv (tprod fun b => f b))
    -/
  · exact hf.hasProd.inv.tprod_eq
    /-
      🎉 no goals
    -/
  · simp [tprod_eq_one_of_not_multipliable hf,
      tprod_eq_one_of_not_multipliable (mt Multipliable.of_inv hf)]


@[to_additive]
theorem tprod_div (hf : Multipliable f) (hg : Multipliable g) :
    ∏' b, (f b / g b) = (∏' b, f b) / ∏' b, g b :=
  (hf.hasProd.div hg.hasProd).tprod_eq


@[to_additive]
theorem prod_mul_tprod_compl {s : Finset β} (hf : Multipliable f) :
    (∏ x ∈ s, f x) * ∏' x : ↑(s : Set β)ᶜ, f x = ∏' x, f x :=
  ((s.hasProd f).mul_compl (s.multipliable_compl_iff.2 hf).hasProd).tprod_eq.symm


/-- Let `f : β → α` be a multipliable function and let `b ∈ β` be an index.
Lemma `tprod_eq_mul_tprod_ite` writes `∏ n, f n` as `f b` times the product of the
remaining terms. -/
@[to_additive "Let `f : β → α` be a summable function and let `b ∈ β` be an index.
Lemma `tsum_eq_add_tsum_ite` writes `Σ' n, f n` as `f b` plus the sum of the
remaining terms."]
theorem tprod_eq_mul_tprod_ite [DecidableEq β] (hf : Multipliable f) (b : β) :
    ∏' n, f n = f b * ∏' n, ite (n = b) 1 (f n) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : CommGroup α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalGroup α
    f : β → α
    inst✝¹ : T2Space α
    inst✝ : DecidableEq β
    hf : Multipliable f
    b : β
    ⊢ Eq (tprod fun n => f n) (HMul.hMul (f b) (tprod fun n => ite (Eq n b) 1 (f n …
  -/
  rw [(hasProd_ite_div_hasProd hf.hasProd b).tprod_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : CommGroup α
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalGroup α
    f : β → α
    inst✝¹ : T2Space α
    inst✝ : DecidableEq β
    hf : Multipliable f
    b : β
    ⊢ Eq (tprod fun n => f n) (HMul.hMul (f b) (HDiv.hDiv (tprod fun b => f b) (f  …
  -/
  exact (mul_div_cancel _ _).symm
  /-
    🎉 no goals
  -/


/-- The **Cauchy criterion** for infinite products, also known as the **Cauchy convergence test** -/
@[to_additive "The **Cauchy criterion** for infinite sums, also known as the
**Cauchy convergence test**"]
theorem multipliable_iff_cauchySeq_finset [CompleteSpace α] {f : β → α} :
    Multipliable f ↔ CauchySeq fun s : Finset β ↦ ∏ b ∈ s, f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : UniformSpace α
    inst✝ : CompleteSpace α
    f : β → α
    ⊢ Iff (Multipliable f) (CauchySeq fun s => s.prod fun b => f b)
  -/
  classical exact cauchy_map_iff_exists_tendsto.symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem cauchySeq_finset_iff_prod_vanishing :
    (CauchySeq fun s : Finset β ↦ ∏ b ∈ s, f b) ↔
      ∀ e ∈ 𝓝 (1 : α), ∃ s : Finset β, ∀ t, Disjoint t s → (∏ b ∈ t, f b) ∈ e := by
  classical
  simp only [CauchySeq, cauchy_map_iff, and_iff_right atTop_neBot, prod_atTop_atTop_eq,
    uniformity_eq_comap_nhds_one α, tendsto_comap_iff, Function.comp_def, atTop_neBot, true_and]
  rw [tendsto_atTop']
  constructor
  · intro h e he
    obtain ⟨⟨s₁, s₂⟩, h⟩ := h e he
    use s₁ ∪ s₂
    intro t ht
    specialize h (s₁ ∪ s₂, s₁ ∪ s₂ ∪ t) ⟨le_sup_left, le_sup_of_le_left le_sup_right⟩
    simpa only [Finset.prod_union ht.symm, mul_div_cancel_left] using h
  · rintro h e he
    rcases exists_nhds_split_inv he with ⟨d, hd, hde⟩
    rcases h d hd with ⟨s, h⟩
    use (s, s)
    rintro ⟨t₁, t₂⟩ ⟨ht₁, ht₂⟩
    have : ((∏ b ∈ t₂, f b) / ∏ b ∈ t₁, f b) = (∏ b ∈ t₂ \ s, f b) / ∏ b ∈ t₁ \ s, f b := by
      rw [← Finset.prod_sdiff ht₁, ← Finset.prod_sdiff ht₂, mul_div_mul_right_eq_div]
    simp only [this]
    exact hde _ (h _ Finset.sdiff_disjoint) _ (h _ Finset.sdiff_disjoint)


@[to_additive]
theorem cauchySeq_finset_iff_tprod_vanishing :
    (CauchySeq fun s : Finset β ↦ ∏ b ∈ s, f b) ↔
      ∀ e ∈ 𝓝 (1 : α), ∃ s : Finset β, ∀ t : Set β, Disjoint t s → (∏' b : t, f b) ∈ e := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : UniformSpace α
    inst✝ : UniformGroup α
    f : β → α
    ⊢ Iff (CauchySeq fun s => s.prod fun b => f b) (∀ (e : Set α), Membership.mem  …
  -/
  simp_rw [cauchySeq_finset_iff_prod_vanishing, Set.disjoint_left, disjoint_left]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommGroup α
    inst✝¹ : UniformSpace α
    inst✝ : UniformGroup α
    f : β → α
    ⊢ Iff (∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Finse …
  -/
  refine ⟨fun vanish e he ↦ ?_, fun vanish e he ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : CommGroup α
      inst✝¹ : UniformSpace α
      inst✝ : UniformGroup α
      f : β → α
      vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Fin …
      e : Set α
      he : Membership.mem (nhds 1) e
      ⊢ Exists fun s => ∀ (t : Set β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Members …
    -/
  · obtain ⟨o, ho, o_closed, oe⟩ := exists_mem_nhds_isClosed_subset he
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : CommGroup α
      inst✝¹ : UniformSpace α
      inst✝ : UniformGroup α
      f : β → α
      vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Fin …
      e : Set α
      he : Membership.mem (nhds 1) e
      o : Set α
      ho : Membership.mem (nhds 1) o
      o_closed : IsClosed o
      oe : HasSubset.Subset o e
      ⊢ Exists fun s => ∀ (t : Set β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Members …
    -/
    obtain ⟨s, hs⟩ := vanish o ho
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : CommGroup α
      inst✝¹ : UniformSpace α
      inst✝ : UniformGroup α
      f : β → α
      vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Fin …
      e : Set α
      he : Membership.mem (nhds 1) e
      o : Set α
      ho : Membership.mem (nhds 1) o
      o_closed : IsClosed o
      oe : HasSubset.Subset o e
      s : Finset β
      hs : ∀ (t : Finset β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem s  …
      ⊢ Exists fun s => ∀ (t : Set β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Members …
    -/
    refine ⟨s, fun t hts ↦ oe ?_⟩
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : CommGroup α
      inst✝¹ : UniformSpace α
      inst✝ : UniformGroup α
      f : β → α
      vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Fin …
      e : Set α
      he : Membership.mem (nhds 1) e
      o : Set α
      ho : Membership.mem (nhds 1) o
      o_closed : IsClosed o
      oe : HasSubset.Subset o e
      s : Finset β
      hs : ∀ (t : Finset β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem s  …
      t : Set β
      hts : ∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem (↑s) a)
      ⊢ Membership.mem o (tprod fun b => f ↑b)
    -/
    by_cases ht : Multipliable fun a : t ↦ f a
    · classical
      refine o_closed.mem_of_tendsto ht.hasProd (Eventually.of_forall fun t' ↦ ?_)
      rw [← prod_subtype_map_embedding fun _ _ ↦ by rfl]
      apply hs
      simp_rw [Finset.mem_map]
      rintro _ ⟨b, -, rfl⟩
      exact hts b.prop
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : CommGroup α
        inst✝¹ : UniformSpace α
        inst✝ : UniformGroup α
        f : β → α
        vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Fin …
        e : Set α
        he : Membership.mem (nhds 1) e
        o : Set α
        ho : Membership.mem (nhds 1) o
        o_closed : IsClosed o
        oe : HasSubset.Subset o e
        s : Finset β
        hs : ∀ (t : Finset β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem s  …
        t : Set β
        hts : ∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem (↑s) a)
        ht : Not (Multipliable fun a => f ↑a)
        ⊢ Membership.mem o (tprod fun b => f ↑b)
      -/
    · exact tprod_eq_one_of_not_multipliable ht ▸ mem_of_mem_nhds ho
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : CommGroup α
      inst✝¹ : UniformSpace α
      inst✝ : UniformGroup α
      f : β → α
      vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Set …
      e : Set α
      he : Membership.mem (nhds 1) e
      ⊢ Exists fun s => ∀ (t : Finset β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Memb …
    -/
  · obtain ⟨s, hs⟩ := vanish _ he
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      inst✝² : CommGroup α
      inst✝¹ : UniformSpace α
      inst✝ : UniformGroup α
      f : β → α
      vanish : ∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Set …
      e : Set α
      he : Membership.mem (nhds 1) e
      s : Finset β
      hs : ∀ (t : Set β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem (↑s)  …
      ⊢ Exists fun s => ∀ (t : Finset β), (∀ ⦃a : β⦄, Membership.mem t a → Not (Memb …
    -/
    exact ⟨s, fun t hts ↦ (t.tprod_subtype f).symm ▸ hs _ hts⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem multipliable_iff_vanishing :
    Multipliable f ↔
    ∀ e ∈ 𝓝 (1 : α), ∃ s : Finset β, ∀ t, Disjoint t s → (∏ b ∈ t, f b) ∈ e := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : CommGroup α
    inst✝² : UniformSpace α
    inst✝¹ : UniformGroup α
    f : β → α
    inst✝ : CompleteSpace α
    ⊢ Iff (Multipliable f) (∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun  …
  -/
  rw [multipliable_iff_cauchySeq_finset, cauchySeq_finset_iff_prod_vanishing]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multipliable_iff_tprod_vanishing : Multipliable f ↔
    ∀ e ∈ 𝓝 (1 : α), ∃ s : Finset β, ∀ t : Set β, Disjoint t s → (∏' b : t, f b) ∈ e := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : CommGroup α
    inst✝² : UniformSpace α
    inst✝¹ : UniformGroup α
    f : β → α
    inst✝ : CompleteSpace α
    ⊢ Iff (Multipliable f) (∀ (e : Set α), Membership.mem (nhds 1) e → Exists fun  …
  -/
  rw [multipliable_iff_cauchySeq_finset, cauchySeq_finset_iff_tprod_vanishing]
  /-
    🎉 no goals
  -/

-- TODO: generalize to monoid with a uniform continuous subtraction operator: `(a + b) - b = a`

@[to_additive]
theorem Multipliable.multipliable_of_eq_one_or_self (hf : Multipliable f)
    (h : ∀ b, g b = 1 ∨ g b = f b) : Multipliable g := by
  classical
  exact multipliable_iff_vanishing.2 fun e he ↦
    let ⟨s, hs⟩ := multipliable_iff_vanishing.1 hf e he
    ⟨s, fun t ht ↦
      have eq : ∏ b ∈ t with g b = f b, f b = ∏ b ∈ t, g b :=
        calc
          ∏ b ∈ t with g b = f b, f b = ∏ b ∈ t with g b = f b, g b :=
            Finset.prod_congr rfl fun b hb ↦ (Finset.mem_filter.1 hb).2.symm
          _ = ∏ b ∈ t, g b := by
           {refine Finset.prod_subset (Finset.filter_subset _ _) ?_
            intro b hbt hb
            simp only [Finset.mem_filter, and_iff_right hbt] at hb
            exact (h b).resolve_right hb}
      eq ▸ hs _ <| Finset.disjoint_of_subset_left (Finset.filter_subset _ _) ht⟩


@[to_additive]
protected theorem Multipliable.mulIndicator (hf : Multipliable f) (s : Set β) :
    Multipliable (s.mulIndicator f) :=
  hf.multipliable_of_eq_one_or_self <| Set.mulIndicator_eq_one_or_self _ _


@[to_additive]
theorem Multipliable.comp_injective {i : γ → β} (hf : Multipliable f) (hi : Injective i) :
    Multipliable (f ∘ i) := by
  simpa only [Set.mulIndicator_range_comp] using
    (hi.multipliable_iff (fun x hx ↦ Set.mulIndicator_of_not_mem hx _)).2
    (hf.mulIndicator (Set.range i))


@[to_additive]
theorem Multipliable.subtype (hf : Multipliable f) (s : Set β) : Multipliable (f ∘ (↑) : s → α) :=
  hf.comp_injective Subtype.coe_injective


@[to_additive]
theorem multipliable_subtype_and_compl {s : Set β} :
    ((Multipliable fun x : s ↦ f x) ∧ Multipliable fun x : ↑sᶜ ↦ f x) ↔ Multipliable f :=
  ⟨and_imp.2 Multipliable.mul_compl, fun h ↦ ⟨h.subtype s, h.subtype sᶜ⟩⟩


@[to_additive]
theorem tprod_subtype_mul_tprod_subtype_compl [T2Space α] {f : β → α} (hf : Multipliable f)
    (s : Set β) : (∏' x : s, f x) * ∏' x : ↑sᶜ, f x = ∏' x, f x :=
  ((hf.subtype s).hasProd.mul_compl (hf.subtype { x | x ∉ s }).hasProd).unique hf.hasProd


@[to_additive]
theorem prod_mul_tprod_subtype_compl [T2Space α] {f : β → α} (hf : Multipliable f) (s : Finset β) :
    (∏ x ∈ s, f x) * ∏' x : { x // x ∉ s }, f x = ∏' x, f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : CommGroup α
    inst✝³ : UniformSpace α
    inst✝² : UniformGroup α
    inst✝¹ : CompleteSpace α
    inst✝ : T2Space α
    f : β → α
    hf : Multipliable f
    s : Finset β
    ⊢ Eq (HMul.hMul (s.prod fun x => f x) (tprod fun x => f ↑x)) (tprod fun x => f …
  -/
  rw [← tprod_subtype_mul_tprod_subtype_compl hf s]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : CommGroup α
    inst✝³ : UniformSpace α
    inst✝² : UniformGroup α
    inst✝¹ : CompleteSpace α
    inst✝ : T2Space α
    f : β → α
    hf : Multipliable f
    s : Finset β
    ⊢ Eq (HMul.hMul (s.prod fun x => f x) (tprod fun x => f ↑x)) (HMul.hMul (tprod …
  -/
  simp only [Finset.tprod_subtype', mul_right_inj]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : CommGroup α
    inst✝³ : UniformSpace α
    inst✝² : UniformGroup α
    inst✝¹ : CompleteSpace α
    inst✝ : T2Space α
    f : β → α
    hf : Multipliable f
    s : Finset β
    ⊢ Eq (tprod fun x => f ↑x) (tprod fun x => f ↑x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multipliable.vanishing (hf : Multipliable f) ⦃e : Set G⦄ (he : e ∈ 𝓝 (1 : G)) :
    ∃ s : Finset α, ∀ t, Disjoint t s → (∏ k ∈ t, f k) ∈ e := by
  classical
  letI : UniformSpace G := TopologicalGroup.toUniformSpace G
  have : UniformGroup G := comm_topologicalGroup_is_uniform
  exact cauchySeq_finset_iff_prod_vanishing.1 hf.hasProd.cauchySeq e he


@[to_additive]
theorem Multipliable.tprod_vanishing (hf : Multipliable f) ⦃e : Set G⦄ (he : e ∈ 𝓝 1) :
    ∃ s : Finset α, ∀ t : Set α, Disjoint t s → (∏' b : t, f b) ∈ e := by
  classical
  letI : UniformSpace G := TopologicalGroup.toUniformSpace G
  have : UniformGroup G := comm_topologicalGroup_is_uniform
  exact cauchySeq_finset_iff_tprod_vanishing.1 hf.hasProd.cauchySeq e he


/-- The product over the complement of a finset tends to `1` when the finset grows to cover the
whole space. This does not need a multipliability assumption, as otherwise all such products are
one. -/
@[to_additive "The sum over the complement of a finset tends to `0` when the finset grows to cover
the whole space. This does not need a summability assumption, as otherwise all such sums are zero."]
theorem tendsto_tprod_compl_atTop_one (f : α → G) :
    Tendsto (fun s : Finset α ↦ ∏' a : { x // x ∉ s }, f a) atTop (𝓝 1) := by
  classical
  by_cases H : Multipliable f
  · intro e he
    obtain ⟨s, hs⟩ := H.tprod_vanishing he
    rw [Filter.mem_map, mem_atTop_sets]
    exact ⟨s, fun t hts ↦ hs _ <| Set.disjoint_left.mpr fun a ha has ↦ ha (hts has)⟩
  · refine tendsto_const_nhds.congr fun _ ↦ (tprod_eq_one_of_not_multipliable ?_).symm
    rwa [Finset.multipliable_compl_iff]


/-- Product divergence test: if `f` is unconditionally multipliable, then `f x` tends to one along
`cofinite`. -/
@[to_additive "Series divergence test: if `f` is unconditionally summable, then `f x` tends to zero
along `cofinite`."]
theorem Multipliable.tendsto_cofinite_one (hf : Multipliable f) : Tendsto f cofinite (𝓝 1) := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : TopologicalSpace G
    inst✝¹ : CommGroup G
    inst✝ : TopologicalGroup G
    f : α → G
    hf : Multipliable f
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 1)
  -/
  intro e he
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : TopologicalSpace G
    inst✝¹ : CommGroup G
    inst✝ : TopologicalGroup G
    f : α → G
    hf : Multipliable f
    e : Set G
    he : Membership.mem (nhds 1) e
    ⊢ Membership.mem (Filter.map f Filter.cofinite) e
  -/
  rw [Filter.mem_map]
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : TopologicalSpace G
    inst✝¹ : CommGroup G
    inst✝ : TopologicalGroup G
    f : α → G
    hf : Multipliable f
    e : Set G
    he : Membership.mem (nhds 1) e
    ⊢ Membership.mem Filter.cofinite (Set.preimage f e)
  -/
  rcases hf.vanishing he with ⟨s, hs⟩
  /-
    case intro
    α : Type u_1
    G : Type u_4
    inst✝² : TopologicalSpace G
    inst✝¹ : CommGroup G
    inst✝ : TopologicalGroup G
    f : α → G
    hf : Multipliable f
    e : Set G
    he : Membership.mem (nhds 1) e
    s : Finset α
    hs : ∀ (t : Finset α), Disjoint t s → Membership.mem e (t.prod fun k => f k)
    ⊢ Membership.mem Filter.cofinite (Set.preimage f e)
  -/
  refine s.eventually_cofinite_nmem.mono fun x hx ↦ ?_
    /-
      case intro
      α : Type u_1
      G : Type u_4
      inst✝² : TopologicalSpace G
      inst✝¹ : CommGroup G
      inst✝ : TopologicalGroup G
      f : α → G
      hf : Multipliable f
      e : Set G
      he : Membership.mem (nhds 1) e
      s : Finset α
      hs : ∀ (t : Finset α), Disjoint t s → Membership.mem e (t.prod fun k => f k)
      x : α
      hx : Not (Membership.mem s x)
      ⊢ Membership.mem e (f x)
    -/
  · simpa using hs {x} (disjoint_singleton_left.2 hx)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem Multipliable.countable_mulSupport [FirstCountableTopology G] [T1Space G]
    (hf : Multipliable f) : f.mulSupport.Countable := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝⁴ : TopologicalSpace G
    inst✝³ : CommGroup G
    inst✝² : TopologicalGroup G
    f : α → G
    inst✝¹ : FirstCountableTopology G
    inst✝ : T1Space G
    hf : Multipliable f
    ⊢ (Function.mulSupport f).Countable
  -/
  simpa only [ker_nhds] using hf.tendsto_cofinite_one.countable_compl_preimage_ker
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multipliable_const_iff [Infinite β] [T2Space G] (a : G) :
    Multipliable (fun _ : β ↦ a) ↔ a = 1 := by
  /-
    β : Type u_2
    G : Type u_4
    inst✝⁴ : TopologicalSpace G
    inst✝³ : CommGroup G
    inst✝² : TopologicalGroup G
    inst✝¹ : Infinite β
    inst✝ : T2Space G
    a : G
    ⊢ Iff (Multipliable fun x => a) (Eq a 1)
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      β : Type u_2
      G : Type u_4
      inst✝⁴ : TopologicalSpace G
      inst✝³ : CommGroup G
      inst✝² : TopologicalGroup G
      inst✝¹ : Infinite β
      inst✝ : T2Space G
      a : G
      h : Multipliable fun x => a
      ⊢ Eq a 1
    -/
  · by_contra ha
    /-
      case refine_1
      β : Type u_2
      G : Type u_4
      inst✝⁴ : TopologicalSpace G
      inst✝³ : CommGroup G
      inst✝² : TopologicalGroup G
      inst✝¹ : Infinite β
      inst✝ : T2Space G
      a : G
      h : Multipliable fun x => a
      ha : Not (Eq a 1)
      ⊢ False
    -/
    have : {a}ᶜ ∈ 𝓝 1 := compl_singleton_mem_nhds (Ne.symm ha)
    have : Finite β := by
      simpa [← Set.finite_univ_iff] using h.tendsto_cofinite_one this
    /-
      case refine_1
      β : Type u_2
      G : Type u_4
      inst✝⁴ : TopologicalSpace G
      inst✝³ : CommGroup G
      inst✝² : TopologicalGroup G
      inst✝¹ : Infinite β
      inst✝ : T2Space G
      a : G
      h : Multipliable fun x => a
      ha : Not (Eq a 1)
      this✝ : Membership.mem (nhds 1) (HasCompl.compl (Singleton.singleton a))
      this : Finite β
      ⊢ False
    -/
    exact not_finite β
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      β : Type u_2
      G : Type u_4
      inst✝⁴ : TopologicalSpace G
      inst✝³ : CommGroup G
      inst✝² : TopologicalGroup G
      inst✝¹ : Infinite β
      inst✝ : T2Space G
      a : G
      ⊢ Eq a 1 → Multipliable fun x => a
    -/
  · rintro rfl
    /-
      case refine_2
      β : Type u_2
      G : Type u_4
      inst✝⁴ : TopologicalSpace G
      inst✝³ : CommGroup G
      inst✝² : TopologicalGroup G
      inst✝¹ : Infinite β
      inst✝ : T2Space G
      ⊢ Multipliable fun x => 1
    -/
    exact multipliable_one
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem tprod_const [T2Space G] (a : G) : ∏' _ : β, a = a ^ (Nat.card β) := by
  /-
    β : Type u_2
    G : Type u_4
    inst✝³ : TopologicalSpace G
    inst✝² : CommGroup G
    inst✝¹ : TopologicalGroup G
    inst✝ : T2Space G
    a : G
    ⊢ Eq (tprod fun x => a) (HPow.hPow a (Nat.card β))
  -/
  rcases finite_or_infinite β with hβ|hβ
    /-
      case inl
      β : Type u_2
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : CommGroup G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      a : G
      hβ : Finite β
      ⊢ Eq (tprod fun x => a) (HPow.hPow a (Nat.card β))
    -/
  · letI : Fintype β := Fintype.ofFinite β
    /-
      case inl
      β : Type u_2
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : CommGroup G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      a : G
      hβ : Finite β
      this : Fintype β := Fintype.ofFinite β
      ⊢ Eq (tprod fun x => a) (HPow.hPow a (Nat.card β))
    -/
    rw [tprod_eq_prod (s := univ) (fun x hx ↦ (hx (mem_univ x)).elim)]
    /-
      case inl
      β : Type u_2
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : CommGroup G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      a : G
      hβ : Finite β
      this : Fintype β := Fintype.ofFinite β
      ⊢ Eq (Finset.univ.prod fun b => a) (HPow.hPow a (Nat.card β))
    -/
    simp only [prod_const, Nat.card_eq_fintype_card, Fintype.card]
    /-
      🎉 no goals
    -/
    /-
      case inr
      β : Type u_2
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : CommGroup G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      a : G
      hβ : Infinite β
      ⊢ Eq (tprod fun x => a) (HPow.hPow a (Nat.card β))
    -/
  · simp only [Nat.card_eq_zero_of_infinite, pow_zero]
    /-
      case inr
      β : Type u_2
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : CommGroup G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      a : G
      hβ : Infinite β
      ⊢ Eq (tprod fun x => a) 1
    -/
    rcases eq_or_ne a 1 with rfl|ha
      /-
        case inr.inl
        β : Type u_2
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : CommGroup G
        inst✝¹ : TopologicalGroup G
        inst✝ : T2Space G
        hβ : Infinite β
        ⊢ Eq (tprod fun x => 1) 1
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        β : Type u_2
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : CommGroup G
        inst✝¹ : TopologicalGroup G
        inst✝ : T2Space G
        a : G
        hβ : Infinite β
        ha : Ne a 1
        ⊢ Eq (tprod fun x => a) 1
      -/
    · apply tprod_eq_one_of_not_multipliable
      /-
        case inr.inr.h
        β : Type u_2
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : CommGroup G
        inst✝¹ : TopologicalGroup G
        inst✝ : T2Space G
        a : G
        hβ : Infinite β
        ha : Ne a 1
        ⊢ Not (Multipliable fun b => a)
      -/
      simpa [multipliable_const_iff] using ha
      /-
        🎉 no goals
      -/


