/-- The set `1 : Set α` is defined as `{1}` in locale `Pointwise`. -/
@[to_additive "The set `0 : Set α` is defined as `{0}` in locale `Pointwise`."]
protected noncomputable def one : One (Set α) :=
  ⟨{1}⟩


@[to_additive]
theorem singleton_one : ({1} : Set α) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem mem_one : a ∈ (1 : Set α) ↔ a = 1 :=
  Iff.rfl


@[to_additive]
theorem one_mem_one : (1 : α) ∈ (1 : Set α) :=
  Eq.refl _


@[to_additive (attr := simp)]
theorem one_subset : 1 ⊆ s ↔ (1 : α) ∈ s :=
  singleton_subset_iff


@[to_additive (attr := simp)]
theorem one_nonempty : (1 : Set α).Nonempty :=
  ⟨1, rfl⟩


@[to_additive (attr := simp)]
theorem image_one {f : α → β} : f '' 1 = {f 1} :=
  image_singleton


@[to_additive]
theorem subset_one_iff_eq : s ⊆ 1 ↔ s = ∅ ∨ s = 1 :=
  subset_singleton_iff_eq


@[to_additive]
theorem Nonempty.subset_one_iff (h : s.Nonempty) : s ⊆ 1 ↔ s = 1 :=
  h.subset_singleton_iff


/-- The singleton operation as a `OneHom`. -/
@[to_additive "The singleton operation as a `ZeroHom`."]
noncomputable def singletonOneHom : OneHom α (Set α) where
  toFun := singleton; map_one' := singleton_one


@[to_additive (attr := simp)]
theorem coe_singletonOneHom : (singletonOneHom : α → Set α) = singleton :=
  rfl


@[to_additive] lemma image_op_one : (1 : Set α).image op = 1 := image_singleton


/-- The pointwise inversion of set `s⁻¹` is defined as `{x | x⁻¹ ∈ s}` in locale `Pointwise`. It is
equal to `{x⁻¹ | x ∈ s}`, see `Set.image_inv_eq_inv`. -/
@[to_additive
      "The pointwise negation of set `-s` is defined as `{x | -x ∈ s}` in locale `Pointwise`.
      It is equal to `{-x | x ∈ s}`, see `Set.image_neg_eq_neg`."]
protected def inv [Inv α] : Inv (Set α) :=
  ⟨preimage Inv.inv⟩


@[to_additive (attr := simp)]
theorem mem_inv : a ∈ s⁻¹ ↔ a⁻¹ ∈ s :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem inv_preimage : Inv.inv ⁻¹' s = s⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem inv_empty : (∅ : Set α)⁻¹ = ∅ :=
  rfl


@[to_additive (attr := simp)]
theorem inv_univ : (univ : Set α)⁻¹ = univ :=
  rfl


@[to_additive (attr := simp)]
theorem inter_inv : (s ∩ t)⁻¹ = s⁻¹ ∩ t⁻¹ :=
  preimage_inter


@[to_additive (attr := simp)]
theorem union_inv : (s ∪ t)⁻¹ = s⁻¹ ∪ t⁻¹ :=
  preimage_union


@[to_additive (attr := simp)]
theorem iInter_inv (s : ι → Set α) : (⋂ i, s i)⁻¹ = ⋂ i, (s i)⁻¹ :=
  preimage_iInter


@[to_additive (attr := simp)]
theorem sInter_inv (S : Set (Set α)) : (⋂₀ S)⁻¹ = ⋂ s ∈ S, s⁻¹ :=
  preimage_sInter


@[to_additive (attr := simp)]
theorem iUnion_inv (s : ι → Set α) : (⋃ i, s i)⁻¹ = ⋃ i, (s i)⁻¹ :=
  preimage_iUnion


@[to_additive (attr := simp)]
theorem sUnion_inv (S : Set (Set α)) : (⋃₀ S)⁻¹ = ⋃ s ∈ S, s⁻¹ :=
  preimage_sUnion


@[to_additive (attr := simp)]
theorem compl_inv : sᶜ⁻¹ = s⁻¹ᶜ :=
  preimage_compl


@[to_additive]
                                              /-
                                                α : Type u_2
                                                inst✝ : InvolutiveInv α
                                                s : Set α
                                                a : α
                                                ⊢ Iff (Membership.mem (Inv.inv s) (Inv.inv a)) (Membership.mem s a)
                                              -/
theorem inv_mem_inv : a⁻¹ ∈ s⁻¹ ↔ a ∈ s := by simp only [mem_inv, inv_inv]
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive (attr := simp)]
theorem nonempty_inv : s⁻¹.Nonempty ↔ s.Nonempty :=
  inv_involutive.surjective.nonempty_preimage


@[to_additive]
theorem Nonempty.inv (h : s.Nonempty) : s⁻¹.Nonempty :=
  nonempty_inv.2 h


@[to_additive (attr := simp)]
theorem image_inv_eq_inv : (·⁻¹) '' s = s⁻¹ :=
  congr_fun (image_eq_preimage_of_inverse inv_involutive.leftInverse inv_involutive.rightInverse) _


@[to_additive (attr := simp)]
theorem inv_eq_empty : s⁻¹ = ∅ ↔ s = ∅ := by
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    s : Set α
    ⊢ Iff (Eq (Inv.inv s) EmptyCollection.emptyCollection) (Eq s EmptyCollection.e …
  -/
  rw [← image_inv_eq_inv, image_eq_empty]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
noncomputable instance involutiveInv : InvolutiveInv (Set α) where
  inv := Inv.inv
                  /-
                    F : Type u_1
                    α : Type u_2
                    β : Type u_3
                    γ : Type u_4
                    inst✝ : InvolutiveInv α
                    s✝ t : Set α
                    a : α
                    s : Set α
                    ⊢ Eq (Inv.inv (Inv.inv s)) s
                  -/
  inv_inv s := by simp only [← inv_preimage, preimage_preimage, inv_inv, preimage_id']
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem inv_subset_inv : s⁻¹ ⊆ t⁻¹ ↔ s ⊆ t :=
  (Equiv.inv α).surjective.preimage_subset_preimage_iff


@[to_additive]
                                             /-
                                               α : Type u_2
                                               inst✝ : InvolutiveInv α
                                               s t : Set α
                                               ⊢ Iff (HasSubset.Subset (Inv.inv s) t) (HasSubset.Subset s (Inv.inv t))
                                             -/
theorem inv_subset : s⁻¹ ⊆ t ↔ s ⊆ t⁻¹ := by rw [← inv_subset_inv, inv_inv]
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive (attr := simp)]
theorem inv_singleton (a : α) : ({a} : Set α)⁻¹ = {a⁻¹} := by
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    a : α
    ⊢ Eq (Inv.inv (Singleton.singleton a)) (Singleton.singleton (Inv.inv a))
  -/
  rw [← image_inv_eq_inv, image_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inv_insert (a : α) (s : Set α) : (insert a s)⁻¹ = insert a⁻¹ s⁻¹ := by
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    a : α
    s : Set α
    ⊢ Eq (Inv.inv (Insert.insert a s)) (Insert.insert (Inv.inv a) (Inv.inv s))
  -/
  rw [insert_eq, union_inv, inv_singleton, insert_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inv_range {ι : Sort*} {f : ι → α} : (range f)⁻¹ = range fun i => (f i)⁻¹ := by
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    ι : Sort u_5
    f : ι → α
    ⊢ Eq (Inv.inv (Set.range f)) (Set.range fun i => Inv.inv (f i))
  -/
  rw [← image_inv_eq_inv]
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    ι : Sort u_5
    f : ι → α
    ⊢ Eq (Set.image (fun x => Inv.inv x) (Set.range f)) (Set.range fun i => Inv.in …
  -/
  exact (range_comp ..).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem image_op_inv : op '' s⁻¹ = (op '' s)⁻¹ := by
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    s : Set α
    ⊢ Eq (Set.image MulOpposite.op (Inv.inv s)) (Inv.inv (Set.image MulOpposite.op …
  -/
  simp_rw [← image_inv_eq_inv, Function.Semiconj.set_image op_inv s]
  /-
    🎉 no goals
  -/


/-- The pointwise multiplication of sets `s * t` and `t` is defined as `{x * y | x ∈ s, y ∈ t}` in
locale `Pointwise`. -/
@[to_additive
      "The pointwise addition of sets `s + t` is defined as `{x + y | x ∈ s, y ∈ t}` in locale
      `Pointwise`."]
protected def mul : Mul (Set α) :=
  ⟨image2 (· * ·)⟩


@[to_additive (attr := simp)]
theorem image2_mul : image2 (· * ·) s t = s * t :=
  rfl


@[to_additive]
theorem mem_mul : a ∈ s * t ↔ ∃ x ∈ s, ∃ y ∈ t, x * y = a :=
  Iff.rfl


@[to_additive]
theorem mul_mem_mul : a ∈ s → b ∈ t → a * b ∈ s * t :=
  mem_image2_of_mem


@[to_additive add_image_prod]
theorem image_mul_prod : (fun x : α × α => x.fst * x.snd) '' s ×ˢ t = s * t :=
  image_prod _


@[to_additive (attr := simp)]
theorem empty_mul : ∅ * s = ∅ :=
  image2_empty_left


@[to_additive (attr := simp)]
theorem mul_empty : s * ∅ = ∅ :=
  image2_empty_right


@[to_additive (attr := simp)]
theorem mul_eq_empty : s * t = ∅ ↔ s = ∅ ∨ t = ∅ :=
  image2_eq_empty_iff


@[to_additive (attr := simp)]
theorem mul_nonempty : (s * t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  image2_nonempty_iff


@[to_additive]
theorem Nonempty.mul : s.Nonempty → t.Nonempty → (s * t).Nonempty :=
  Nonempty.image2


@[to_additive]
theorem Nonempty.of_mul_left : (s * t).Nonempty → s.Nonempty :=
  Nonempty.of_image2_left


@[to_additive]
theorem Nonempty.of_mul_right : (s * t).Nonempty → t.Nonempty :=
  Nonempty.of_image2_right


@[to_additive (attr := simp)]
theorem mul_singleton : s * {b} = (· * b) '' s :=
  image2_singleton_right


@[to_additive (attr := simp)]
theorem singleton_mul : {a} * t = (a * ·) '' t :=
  image2_singleton_left


@[to_additive]
theorem singleton_mul_singleton : ({a} : Set α) * {b} = {a * b} :=
  image2_singleton


@[to_additive (attr := mono, gcongr)]
theorem mul_subset_mul : s₁ ⊆ t₁ → s₂ ⊆ t₂ → s₁ * s₂ ⊆ t₁ * t₂ :=
  image2_subset


@[to_additive (attr := gcongr)]
theorem mul_subset_mul_left : t₁ ⊆ t₂ → s * t₁ ⊆ s * t₂ :=
  image2_subset_left


@[to_additive (attr := gcongr)]
theorem mul_subset_mul_right : s₁ ⊆ s₂ → s₁ * t ⊆ s₂ * t :=
  image2_subset_right


@[to_additive] instance : MulLeftMono (Set α) where elim _s _t₁ _t₂ := mul_subset_mul_left

@[to_additive] instance : MulRightMono (Set α) where elim _t _s₁ _s₂ := mul_subset_mul_right


@[to_additive]
theorem mul_subset_iff : s * t ⊆ u ↔ ∀ x ∈ s, ∀ y ∈ t, x * y ∈ u :=
  image2_subset_iff


@[to_additive]
theorem union_mul : (s₁ ∪ s₂) * t = s₁ * t ∪ s₂ * t :=
  image2_union_left


@[to_additive]
theorem mul_union : s * (t₁ ∪ t₂) = s * t₁ ∪ s * t₂ :=
  image2_union_right


@[to_additive]
theorem inter_mul_subset : s₁ ∩ s₂ * t ⊆ s₁ * t ∩ (s₂ * t) :=
  image2_inter_subset_left


@[to_additive]
theorem mul_inter_subset : s * (t₁ ∩ t₂) ⊆ s * t₁ ∩ (s * t₂) :=
  image2_inter_subset_right


@[to_additive]
theorem inter_mul_union_subset_union : s₁ ∩ s₂ * (t₁ ∪ t₂) ⊆ s₁ * t₁ ∪ s₂ * t₂ :=
  image2_inter_union_subset_union


@[to_additive]
theorem union_mul_inter_subset_union : (s₁ ∪ s₂) * (t₁ ∩ t₂) ⊆ s₁ * t₁ ∪ s₂ * t₂ :=
  image2_union_inter_subset_union


@[to_additive]
theorem iUnion_mul_left_image : ⋃ a ∈ s, (a * ·) '' t = s * t :=
  iUnion_image_left _


@[to_additive]
theorem iUnion_mul_right_image : ⋃ a ∈ t, (· * a) '' s = s * t :=
  iUnion_image_right _


@[to_additive]
theorem iUnion_mul (s : ι → Set α) (t : Set α) : (⋃ i, s i) * t = ⋃ i, s i * t :=
  image2_iUnion_left ..


@[to_additive]
theorem mul_iUnion (s : Set α) (t : ι → Set α) : (s * ⋃ i, t i) = ⋃ i, s * t i :=
  image2_iUnion_right ..


@[to_additive]
theorem sUnion_mul (S : Set (Set α)) (t : Set α) : ⋃₀ S * t = ⋃ s ∈ S, s * t :=
  image2_sUnion_left ..


@[to_additive]
theorem mul_sUnion (s : Set α) (T : Set (Set α)) : s * ⋃₀ T = ⋃ t ∈ T, s * t :=
  image2_sUnion_right ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem iUnion₂_mul (s : ∀ i, κ i → Set α) (t : Set α) :
    (⋃ (i) (j), s i j) * t = ⋃ (i) (j), s i j * t :=
  image2_iUnion₂_left ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem mul_iUnion₂ (s : Set α) (t : ∀ i, κ i → Set α) :
    (s * ⋃ (i) (j), t i j) = ⋃ (i) (j), s * t i j :=
  image2_iUnion₂_right ..


@[to_additive]
theorem iInter_mul_subset (s : ι → Set α) (t : Set α) : (⋂ i, s i) * t ⊆ ⋂ i, s i * t :=
  Set.image2_iInter_subset_left ..


@[to_additive]
theorem mul_iInter_subset (s : Set α) (t : ι → Set α) : (s * ⋂ i, t i) ⊆ ⋂ i, s * t i :=
  image2_iInter_subset_right ..


@[to_additive]
lemma mul_sInter_subset (s : Set α) (T : Set (Set α)) :
    s * ⋂₀ T ⊆ ⋂ t ∈ T, s * t := image2_sInter_right_subset s T (fun a b => a * b)


@[to_additive]
lemma sInter_mul_subset (S : Set (Set α)) (t : Set α) :
    ⋂₀ S * t ⊆ ⋂ s ∈ S, s * t := image2_sInter_left_subset S t (fun a b => a * b)

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem iInter₂_mul_subset (s : ∀ i, κ i → Set α) (t : Set α) :
    (⋂ (i) (j), s i j) * t ⊆ ⋂ (i) (j), s i j * t :=
  image2_iInter₂_subset_left ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem mul_iInter₂_subset (s : Set α) (t : ∀ i, κ i → Set α) :
    (s * ⋂ (i) (j), t i j) ⊆ ⋂ (i) (j), s * t i j :=
  image2_iInter₂_subset_right ..


/-- The singleton operation as a `MulHom`. -/
@[to_additive "The singleton operation as an `AddHom`."]
noncomputable def singletonMulHom : α →ₙ* Set α where
  toFun := singleton
  map_mul' _ _ := singleton_mul_singleton.symm


@[to_additive (attr := simp)]
theorem coe_singletonMulHom : (singletonMulHom : α → Set α) = singleton :=
  rfl


@[to_additive (attr := simp)]
theorem singletonMulHom_apply (a : α) : singletonMulHom a = {a} :=
  rfl


@[to_additive (attr := simp)]
theorem image_op_mul : op '' (s * t) = op '' t * op '' s :=
  image_image2_antidistrib op_mul


/-- The pointwise division of sets `s / t` is defined as `{x / y | x ∈ s, y ∈ t}` in locale
`Pointwise`. -/
@[to_additive
      "The pointwise subtraction of sets `s - t` is defined as `{x - y | x ∈ s, y ∈ t}` in locale
      `Pointwise`."]
protected def div : Div (Set α) :=
  ⟨image2 (· / ·)⟩


@[to_additive (attr := simp)]
theorem image2_div : image2 Div.div s t = s / t :=
  rfl


@[to_additive]
theorem mem_div : a ∈ s / t ↔ ∃ x ∈ s, ∃ y ∈ t, x / y = a :=
  Iff.rfl


@[to_additive]
theorem div_mem_div : a ∈ s → b ∈ t → a / b ∈ s / t :=
  mem_image2_of_mem


@[to_additive sub_image_prod]
theorem image_div_prod : (fun x : α × α => x.fst / x.snd) '' s ×ˢ t = s / t :=
  image_prod _


@[to_additive (attr := simp)]
theorem empty_div : ∅ / s = ∅ :=
  image2_empty_left


@[to_additive (attr := simp)]
theorem div_empty : s / ∅ = ∅ :=
  image2_empty_right


@[to_additive (attr := simp)]
theorem div_eq_empty : s / t = ∅ ↔ s = ∅ ∨ t = ∅ :=
  image2_eq_empty_iff


@[to_additive (attr := simp)]
theorem div_nonempty : (s / t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  image2_nonempty_iff


@[to_additive]
theorem Nonempty.div : s.Nonempty → t.Nonempty → (s / t).Nonempty :=
  Nonempty.image2


@[to_additive]
theorem Nonempty.of_div_left : (s / t).Nonempty → s.Nonempty :=
  Nonempty.of_image2_left


@[to_additive]
theorem Nonempty.of_div_right : (s / t).Nonempty → t.Nonempty :=
  Nonempty.of_image2_right


@[to_additive (attr := simp)]
theorem div_singleton : s / {b} = (· / b) '' s :=
  image2_singleton_right


@[to_additive (attr := simp)]
theorem singleton_div : {a} / t = (· / ·) a '' t :=
  image2_singleton_left


@[to_additive]
theorem singleton_div_singleton : ({a} : Set α) / {b} = {a / b} :=
  image2_singleton


@[to_additive (attr := mono, gcongr)]
theorem div_subset_div : s₁ ⊆ t₁ → s₂ ⊆ t₂ → s₁ / s₂ ⊆ t₁ / t₂ :=
  image2_subset


@[to_additive (attr := gcongr)]
theorem div_subset_div_left : t₁ ⊆ t₂ → s / t₁ ⊆ s / t₂ :=
  image2_subset_left


@[to_additive (attr := gcongr)]
theorem div_subset_div_right : s₁ ⊆ s₂ → s₁ / t ⊆ s₂ / t :=
  image2_subset_right


@[to_additive]
theorem div_subset_iff : s / t ⊆ u ↔ ∀ x ∈ s, ∀ y ∈ t, x / y ∈ u :=
  image2_subset_iff


@[to_additive]
theorem union_div : (s₁ ∪ s₂) / t = s₁ / t ∪ s₂ / t :=
  image2_union_left


@[to_additive]
theorem div_union : s / (t₁ ∪ t₂) = s / t₁ ∪ s / t₂ :=
  image2_union_right


@[to_additive]
theorem inter_div_subset : s₁ ∩ s₂ / t ⊆ s₁ / t ∩ (s₂ / t) :=
  image2_inter_subset_left


@[to_additive]
theorem div_inter_subset : s / (t₁ ∩ t₂) ⊆ s / t₁ ∩ (s / t₂) :=
  image2_inter_subset_right


@[to_additive]
theorem inter_div_union_subset_union : s₁ ∩ s₂ / (t₁ ∪ t₂) ⊆ s₁ / t₁ ∪ s₂ / t₂ :=
  image2_inter_union_subset_union


@[to_additive]
theorem union_div_inter_subset_union : (s₁ ∪ s₂) / (t₁ ∩ t₂) ⊆ s₁ / t₁ ∪ s₂ / t₂ :=
  image2_union_inter_subset_union


@[to_additive]
theorem iUnion_div_left_image : ⋃ a ∈ s, (a / ·) '' t = s / t :=
  iUnion_image_left _


@[to_additive]
theorem iUnion_div_right_image : ⋃ a ∈ t, (· / a) '' s = s / t :=
  iUnion_image_right _


@[to_additive]
theorem iUnion_div (s : ι → Set α) (t : Set α) : (⋃ i, s i) / t = ⋃ i, s i / t :=
  image2_iUnion_left ..


@[to_additive]
theorem div_iUnion (s : Set α) (t : ι → Set α) : (s / ⋃ i, t i) = ⋃ i, s / t i :=
  image2_iUnion_right ..


@[to_additive]
theorem sUnion_div (S : Set (Set α)) (t : Set α) : ⋃₀ S / t = ⋃ s ∈ S, s / t :=
  image2_sUnion_left ..


@[to_additive]
theorem div_sUnion (s : Set α) (T : Set (Set α)) : s / ⋃₀ T = ⋃ t ∈ T, s / t :=
  image2_sUnion_right ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem iUnion₂_div (s : ∀ i, κ i → Set α) (t : Set α) :
    (⋃ (i) (j), s i j) / t = ⋃ (i) (j), s i j / t :=
  image2_iUnion₂_left ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem div_iUnion₂ (s : Set α) (t : ∀ i, κ i → Set α) :
    (s / ⋃ (i) (j), t i j) = ⋃ (i) (j), s / t i j :=
  image2_iUnion₂_right ..


@[to_additive]
theorem iInter_div_subset (s : ι → Set α) (t : Set α) : (⋂ i, s i) / t ⊆ ⋂ i, s i / t :=
  image2_iInter_subset_left ..


@[to_additive]
theorem div_iInter_subset (s : Set α) (t : ι → Set α) : (s / ⋂ i, t i) ⊆ ⋂ i, s / t i :=
  image2_iInter_subset_right ..


@[to_additive]
theorem sInter_div_subset (S : Set (Set α)) (t : Set α) : ⋂₀ S / t ⊆ ⋂ s ∈ S, s / t :=
  image2_sInter_subset_left ..


@[to_additive]
theorem div_sInter_subset (s : Set α) (T : Set (Set α)) : s / ⋂₀ T ⊆ ⋂ t ∈ T, s / t :=
  image2_sInter_subset_right ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem iInter₂_div_subset (s : ∀ i, κ i → Set α) (t : Set α) :
    (⋂ (i) (j), s i j) / t ⊆ ⋂ (i) (j), s i j / t :=
  image2_iInter₂_subset_left ..

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/
/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[to_additive]
theorem div_iInter₂_subset (s : Set α) (t : ∀ i, κ i → Set α) :
    (s / ⋂ (i) (j), t i j) ⊆ ⋂ (i) (j), s / t i j :=
  image2_iInter₂_subset_right ..


/-- The dilation of set `x • s` is defined as `{x • y | y ∈ s}` in locale `Pointwise`. -/
@[to_additive
"The translation of set `x +ᵥ s` is defined as `{x +ᵥ y | y ∈ s}` in locale `Pointwise`."]
protected def smulSet [SMul α β] : SMul α (Set β) where smul a := image (a • ·)


/-- The pointwise scalar multiplication of sets `s • t` is defined as `{x • y | x ∈ s, y ∈ t}` in
locale `Pointwise`. -/
@[to_additive
"The pointwise scalar addition of sets `s +ᵥ t` is defined as `{x +ᵥ y | x ∈ s, y ∈ t}` in locale
`Pointwise`."]
protected def smul [SMul α β] : SMul (Set α) (Set β) where smul := image2 (· • ·)


@[to_additive (attr := simp)] lemma image2_smul : image2 SMul.smul s t = s • t := rfl


@[to_additive vadd_image_prod]
lemma image_smul_prod : (fun x : α × β ↦ x.fst • x.snd) '' s ×ˢ t = s • t := image_prod _


@[to_additive] lemma mem_smul : b ∈ s • t ↔ ∃ x ∈ s, ∃ y ∈ t, x • y = b := Iff.rfl


@[to_additive] lemma smul_mem_smul : a ∈ s → b ∈ t → a • b ∈ s • t := mem_image2_of_mem


@[to_additive (attr := simp)] lemma empty_smul : (∅ : Set α) • t = ∅ := image2_empty_left

@[to_additive (attr := simp)] lemma smul_empty : s • (∅ : Set β) = ∅ := image2_empty_right


@[to_additive (attr := simp)] lemma smul_eq_empty : s • t = ∅ ↔ s = ∅ ∨ t = ∅ := image2_eq_empty_iff


@[to_additive (attr := simp)]
lemma smul_nonempty : (s • t).Nonempty ↔ s.Nonempty ∧ t.Nonempty := image2_nonempty_iff


@[to_additive] lemma Nonempty.smul : s.Nonempty → t.Nonempty → (s • t).Nonempty := .image2

@[to_additive] lemma Nonempty.of_smul_left : (s • t).Nonempty → s.Nonempty := .of_image2_left

@[to_additive] lemma Nonempty.of_smul_right : (s • t).Nonempty → t.Nonempty := .of_image2_right


@[to_additive (attr := simp low+1)]
lemma smul_singleton : s • ({b} : Set β) = (· • b) '' s := image2_singleton_right


@[to_additive (attr := simp low+1)]
lemma singleton_smul : ({a} : Set α) • t = a • t := image2_singleton_left


@[to_additive (attr := simp high)]
lemma singleton_smul_singleton : ({a} : Set α) • ({b} : Set β) = {a • b} := image2_singleton


@[to_additive (attr := mono, gcongr)]
lemma smul_subset_smul : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ • t₁ ⊆ s₂ • t₂ := image2_subset


@[to_additive (attr := gcongr)]
lemma smul_subset_smul_left : t₁ ⊆ t₂ → s • t₁ ⊆ s • t₂ := image2_subset_left


@[to_additive (attr := gcongr)]
lemma smul_subset_smul_right : s₁ ⊆ s₂ → s₁ • t ⊆ s₂ • t := image2_subset_right


@[to_additive] lemma smul_subset_iff : s • t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, a • b ∈ u := image2_subset_iff


@[to_additive] lemma union_smul : (s₁ ∪ s₂) • t = s₁ • t ∪ s₂ • t := image2_union_left

@[to_additive] lemma smul_union : s • (t₁ ∪ t₂) = s • t₁ ∪ s • t₂ := image2_union_right


@[to_additive]
lemma inter_smul_subset : (s₁ ∩ s₂) • t ⊆ s₁ • t ∩ s₂ • t := image2_inter_subset_left


@[to_additive]
lemma smul_inter_subset : s • (t₁ ∩ t₂) ⊆ s • t₁ ∩ s • t₂ := image2_inter_subset_right


@[to_additive]
lemma inter_smul_union_subset_union : (s₁ ∩ s₂) • (t₁ ∪ t₂) ⊆ s₁ • t₁ ∪ s₂ • t₂ :=
  image2_inter_union_subset_union


@[to_additive]
lemma union_smul_inter_subset_union : (s₁ ∪ s₂) • (t₁ ∩ t₂) ⊆ s₁ • t₁ ∪ s₂ • t₂ :=
  image2_union_inter_subset_union


@[to_additive] lemma iUnion_smul_left_image : ⋃ a ∈ s, a • t = s • t := iUnion_image_left _


@[to_additive]
lemma iUnion_smul_right_image : ⋃ a ∈ t, (· • a) '' s = s • t := iUnion_image_right _


@[to_additive]
lemma iUnion_smul (s : ι → Set α) (t : Set β) : (⋃ i, s i) • t = ⋃ i, s i • t :=
  image2_iUnion_left ..


@[to_additive]
lemma smul_iUnion (s : Set α) (t : ι → Set β) : (s • ⋃ i, t i) = ⋃ i, s • t i :=
  image2_iUnion_right ..


@[to_additive]
lemma sUnion_smul (S : Set (Set α)) (t : Set β) : ⋃₀ S • t = ⋃ s ∈ S, s • t :=
  image2_sUnion_left ..


@[to_additive]
lemma smul_sUnion (s : Set α) (T : Set (Set β)) : s • ⋃₀ T = ⋃ t ∈ T, s • t :=
  image2_sUnion_right ..


@[to_additive]
lemma iUnion₂_smul (s : ∀ i, κ i → Set α) (t : Set β) :
    (⋃ i, ⋃ j, s i j) • t = ⋃ i, ⋃ j, s i j • t := image2_iUnion₂_left ..


@[to_additive]
lemma smul_iUnion₂ (s : Set α) (t : ∀ i, κ i → Set β) :
    (s • ⋃ i, ⋃ j, t i j) = ⋃ i, ⋃ j, s • t i j := image2_iUnion₂_right ..


@[to_additive]
lemma iInter_smul_subset (s : ι → Set α) (t : Set β) : (⋂ i, s i) • t ⊆ ⋂ i, s i • t :=
  image2_iInter_subset_left ..


@[to_additive]
lemma smul_iInter_subset (s : Set α) (t : ι → Set β) : (s • ⋂ i, t i) ⊆ ⋂ i, s • t i :=
  image2_iInter_subset_right ..


@[to_additive]
lemma sInter_smul_subset (S : Set (Set α)) (t : Set β) : ⋂₀ S • t ⊆ ⋂ s ∈ S, s • t :=
  image2_sInter_left_subset S t (fun a x => a • x)


@[to_additive]
lemma smul_sInter_subset (s : Set α) (T : Set (Set β)) : s • ⋂₀ T ⊆ ⋂ t ∈ T, s • t :=
  image2_sInter_right_subset s T (fun a x => a • x)


@[to_additive]
lemma iInter₂_smul_subset (s : ∀ i, κ i → Set α) (t : Set β) :
    (⋂ i, ⋂ j, s i j) • t ⊆ ⋂ i, ⋂ j, s i j • t := image2_iInter₂_subset_left ..


@[to_additive]
lemma smul_iInter₂_subset (s : Set α) (t : ∀ i, κ i → Set β) :
    (s • ⋂ i, ⋂ j, t i j) ⊆ ⋂ i, ⋂ j, s • t i j := image2_iInter₂_subset_right ..


@[to_additive]
lemma smul_set_subset_smul {s : Set α} : a ∈ s → a • t ⊆ s • t := image_subset_image2_right


@[to_additive (attr := simp)]
lemma iUnion_smul_set (s : Set α) (t : Set β) : ⋃ a ∈ s, a • t = s • t := iUnion_image_left _


@[to_additive] lemma image_smul : (fun x ↦ a • x) '' t = a • t := rfl


@[to_additive] lemma mem_smul_set : x ∈ a • t ↔ ∃ y, y ∈ t ∧ a • y = x := Iff.rfl


@[to_additive] lemma smul_mem_smul_set : b ∈ s → a • b ∈ a • s := mem_image_of_mem _


@[to_additive (attr := simp)] lemma smul_set_empty : a • (∅ : Set β) = ∅ := image_empty _

@[to_additive (attr := simp)] lemma smul_set_eq_empty : a • s = ∅ ↔ s = ∅ := image_eq_empty


@[to_additive (attr := simp)]
lemma smul_set_nonempty : (a • s).Nonempty ↔ s.Nonempty := image_nonempty


@[to_additive (attr := simp)]
lemma smul_set_singleton : a • ({b} : Set β) = {a • b} := image_singleton


@[to_additive (attr := gcongr)] lemma smul_set_mono : s ⊆ t → a • s ⊆ a • t := image_subset _


@[to_additive]
lemma smul_set_subset_iff : a • s ⊆ t ↔ ∀ ⦃b⦄, b ∈ s → a • b ∈ t :=
  image_subset_iff


@[to_additive]
lemma smul_set_union : a • (t₁ ∪ t₂) = a • t₁ ∪ a • t₂ :=
  image_union ..


@[to_additive]
lemma smul_set_insert (a : α) (b : β) (s : Set β) : a • insert b s = insert (a • b) (a • s) :=
  image_insert_eq ..


@[to_additive]
lemma smul_set_inter_subset : a • (t₁ ∩ t₂) ⊆ a • t₁ ∩ a • t₂ :=
  image_inter_subset ..


@[to_additive]
lemma smul_set_iUnion (a : α) (s : ι → Set β) : a • ⋃ i, s i = ⋃ i, a • s i :=
  image_iUnion


@[to_additive]
lemma smul_set_iUnion₂ (a : α) (s : ∀ i, κ i → Set β) :
    a • ⋃ i, ⋃ j, s i j = ⋃ i, ⋃ j, a • s i j := image_iUnion₂ ..


@[to_additive]
lemma smul_set_sUnion (a : α) (S : Set (Set β)) : a • ⋃₀ S = ⋃ s ∈ S, a • s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SMul α β
    a : α
    S : Set (Set β)
    ⊢ Eq (HSMul.hSMul a S.sUnion) (Set.iUnion fun s => Set.iUnion fun h => HSMul.h …
  -/
  rw [sUnion_eq_biUnion, smul_set_iUnion₂]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma smul_set_iInter_subset (a : α) (t : ι → Set β) : a • ⋂ i, t i ⊆ ⋂ i, a • t i :=
  image_iInter_subset ..


@[to_additive]
lemma smul_set_sInter_subset (a : α) (S : Set (Set β)) :
    a • ⋂₀ S ⊆ ⋂ s ∈ S, a • s := image_sInter_subset ..


@[to_additive]
lemma smul_set_iInter₂_subset (a : α) (t : ∀ i, κ i → Set β) :
    a • ⋂ i, ⋂ j, t i j ⊆ ⋂ i, ⋂ j, a • t i j := image_iInter₂_subset ..


@[to_additive] lemma Nonempty.smul_set : s.Nonempty → (a • s).Nonempty := Nonempty.image _


@[to_additive]
lemma range_smul_range {ι κ : Type*} [SMul α β] (b : ι → α) (c : κ → β) :
    range b • range c = range fun p : ι × κ ↦ b p.1 • c p.2 :=
  image2_range ..


@[to_additive]
lemma smul_set_range [SMul α β] {ι : Sort*} (a : α) (f : ι → β) :
    a • range f = range fun i ↦ a • f i :=
  (range_comp ..).symm


@[to_additive] lemma range_smul [SMul α β] {ι : Sort*} (a : α) (f : ι → β) :
    range (fun i ↦ a • f i) = a • range f := (smul_set_range ..).symm


instance vsub : VSub (Set α) (Set β) where vsub := image2 (· -ᵥ ·)


@[simp] lemma image2_vsub : (image2 VSub.vsub s t : Set α) = s -ᵥ t := rfl


lemma image_vsub_prod : (fun x : β × β ↦ x.fst -ᵥ x.snd) '' s ×ˢ t = s -ᵥ t := image_prod _


lemma mem_vsub : a ∈ s -ᵥ t ↔ ∃ x ∈ s, ∃ y ∈ t, x -ᵥ y = a := Iff.rfl


lemma vsub_mem_vsub (hb : b ∈ s) (hc : c ∈ t) : b -ᵥ c ∈ s -ᵥ t := mem_image2_of_mem hb hc


@[simp] lemma empty_vsub (t : Set β) : ∅ -ᵥ t = ∅ := image2_empty_left

@[simp] lemma vsub_empty (s : Set β) : s -ᵥ ∅ = ∅ := image2_empty_right


@[simp] lemma vsub_eq_empty : s -ᵥ t = ∅ ↔ s = ∅ ∨ t = ∅ := image2_eq_empty_iff


@[simp]
lemma vsub_nonempty : (s -ᵥ t : Set α).Nonempty ↔ s.Nonempty ∧ t.Nonempty := image2_nonempty_iff


lemma Nonempty.vsub : s.Nonempty → t.Nonempty → (s -ᵥ t : Set α).Nonempty := .image2

lemma Nonempty.of_vsub_left : (s -ᵥ t : Set α).Nonempty → s.Nonempty := .of_image2_left

lemma Nonempty.of_vsub_right : (s -ᵥ t : Set α).Nonempty → t.Nonempty := .of_image2_right


@[simp low+1]
lemma vsub_singleton (s : Set β) (b : β) : s -ᵥ {b} = (· -ᵥ b) '' s := image2_singleton_right


@[simp low+1]
lemma singleton_vsub (t : Set β) (b : β) : {b} -ᵥ t = (b -ᵥ ·) '' t := image2_singleton_left


@[simp high] lemma singleton_vsub_singleton : ({b} : Set β) -ᵥ {c} = {b -ᵥ c} := image2_singleton


@[mono, gcongr] lemma vsub_subset_vsub : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ -ᵥ t₁ ⊆ s₂ -ᵥ t₂ := image2_subset


@[gcongr] lemma vsub_subset_vsub_left : t₁ ⊆ t₂ → s -ᵥ t₁ ⊆ s -ᵥ t₂ := image2_subset_left

@[gcongr] lemma vsub_subset_vsub_right : s₁ ⊆ s₂ → s₁ -ᵥ t ⊆ s₂ -ᵥ t := image2_subset_right


lemma vsub_subset_iff : s -ᵥ t ⊆ u ↔ ∀ x ∈ s, ∀ y ∈ t, x -ᵥ y ∈ u := image2_subset_iff


lemma vsub_self_mono (h : s ⊆ t) : s -ᵥ s ⊆ t -ᵥ t := vsub_subset_vsub h h


lemma union_vsub : s₁ ∪ s₂ -ᵥ t = s₁ -ᵥ t ∪ (s₂ -ᵥ t) := image2_union_left

lemma vsub_union : s -ᵥ (t₁ ∪ t₂) = s -ᵥ t₁ ∪ (s -ᵥ t₂) := image2_union_right


lemma inter_vsub_subset : s₁ ∩ s₂ -ᵥ t ⊆ (s₁ -ᵥ t) ∩ (s₂ -ᵥ t) := image2_inter_subset_left

lemma vsub_inter_subset : s -ᵥ t₁ ∩ t₂ ⊆ (s -ᵥ t₁) ∩ (s -ᵥ t₂) := image2_inter_subset_right


lemma inter_vsub_union_subset_union : s₁ ∩ s₂ -ᵥ (t₁ ∪ t₂) ⊆ s₁ -ᵥ t₁ ∪ (s₂ -ᵥ t₂) :=
  image2_inter_union_subset_union


lemma union_vsub_inter_subset_union : s₁ ∪ s₂ -ᵥ t₁ ∩ t₂ ⊆ s₁ -ᵥ t₁ ∪ (s₂ -ᵥ t₂) :=
  image2_union_inter_subset_union


lemma iUnion_vsub_left_image : ⋃ a ∈ s, (a -ᵥ ·) '' t = s -ᵥ t := iUnion_image_left _

lemma iUnion_vsub_right_image : ⋃ a ∈ t, (· -ᵥ a) '' s = s -ᵥ t := iUnion_image_right _


lemma iUnion_vsub (s : ι → Set β) (t : Set β) : (⋃ i, s i) -ᵥ t = ⋃ i, s i -ᵥ t :=
  image2_iUnion_left ..


lemma vsub_iUnion (s : Set β) (t : ι → Set β) : (s -ᵥ ⋃ i, t i) = ⋃ i, s -ᵥ t i :=
  image2_iUnion_right ..


lemma sUnion_vsub (S : Set (Set β)) (t : Set β) : ⋃₀ S -ᵥ t = ⋃ s ∈ S, s -ᵥ t :=
  image2_sUnion_left ..


lemma vsub_sUnion (s : Set β) (T : Set (Set β)) : s -ᵥ ⋃₀ T = ⋃ t ∈ T, s -ᵥ t :=
  image2_sUnion_right ..


lemma iUnion₂_vsub (s : ∀ i, κ i → Set β) (t : Set β) :
    (⋃ i, ⋃ j, s i j) -ᵥ t = ⋃ i, ⋃ j, s i j -ᵥ t := image2_iUnion₂_left ..


lemma vsub_iUnion₂ (s : Set β) (t : ∀ i, κ i → Set β) :
    (s -ᵥ ⋃ i, ⋃ j, t i j) = ⋃ i, ⋃ j, s -ᵥ t i j := image2_iUnion₂_right ..


lemma iInter_vsub_subset (s : ι → Set β) (t : Set β) : (⋂ i, s i) -ᵥ t ⊆ ⋂ i, s i -ᵥ t :=
  image2_iInter_subset_left ..


lemma vsub_iInter_subset (s : Set β) (t : ι → Set β) : (s -ᵥ ⋂ i, t i) ⊆ ⋂ i, s -ᵥ t i :=
  image2_iInter_subset_right ..


lemma sInter_vsub_subset (S : Set (Set β)) (t : Set β) : ⋂₀ S -ᵥ t ⊆ ⋂ s ∈ S, s -ᵥ t :=
  image2_sInter_subset_left ..


lemma vsub_sInter_subset (s : Set β) (T : Set (Set β)) : s -ᵥ ⋂₀ T ⊆ ⋂ t ∈ T, s -ᵥ t :=
  image2_sInter_subset_right ..


lemma iInter₂_vsub_subset (s : ∀ i, κ i → Set β) (t : Set β) :
    (⋂ i, ⋂ j, s i j) -ᵥ t ⊆ ⋂ i, ⋂ j, s i j -ᵥ t := image2_iInter₂_subset_left ..


lemma vsub_iInter₂_subset (s : Set β) (t : ∀ i, κ i → Set β) :
    s -ᵥ ⋂ i, ⋂ j, t i j ⊆ ⋂ i, ⋂ j, s -ᵥ t i j := image2_iInter₂_subset_right ..


@[to_additive]
lemma image_smul_comm [SMul α β] [SMul α γ] (f : β → γ) (a : α) (s : Set β) :
    (∀ b, f (a • b) = a • f b) → f '' (a • s) = a • f '' s := image_comm


/-- Repeated pointwise addition (not the same as pointwise repeated addition!) of a `Set`. See
note [pointwise nat action]. -/
protected def NSMul [Zero α] [Add α] : SMul ℕ (Set α) :=
  ⟨nsmulRec⟩


/-- Repeated pointwise multiplication (not the same as pointwise repeated multiplication!) of a
`Set`. See note [pointwise nat action]. -/
@[to_additive existing]
protected def NPow [One α] [Mul α] : Pow (Set α) ℕ :=
  ⟨fun s n => npowRec n s⟩


/-- Repeated pointwise addition/subtraction (not the same as pointwise repeated
addition/subtraction!) of a `Set`. See note [pointwise nat action]. -/
protected def ZSMul [Zero α] [Add α] [Neg α] : SMul ℤ (Set α) :=
  ⟨zsmulRec⟩


/-- Repeated pointwise multiplication/division (not the same as pointwise repeated
multiplication/division!) of a `Set`. See note [pointwise nat action]. -/
@[to_additive existing]
protected def ZPow [One α] [Mul α] [Inv α] : Pow (Set α) ℤ :=
  ⟨fun s n => zpowRec npowRec n s⟩


/-- `Set α` is a `Semigroup` under pointwise operations if `α` is. -/
@[to_additive "`Set α` is an `AddSemigroup` under pointwise operations if `α` is."]
protected noncomputable def semigroup [Semigroup α] : Semigroup (Set α) :=
  { Set.mul with mul_assoc := fun _ _ _ => image2_assoc mul_assoc }


/-- `Set α` is a `CommSemigroup` under pointwise operations if `α` is. -/
@[to_additive "`Set α` is an `AddCommSemigroup` under pointwise operations if `α` is."]
protected noncomputable def commSemigroup : CommSemigroup (Set α) :=
  { Set.semigroup with mul_comm := fun _ _ => image2_comm mul_comm }


@[to_additive]
theorem inter_mul_union_subset : s ∩ t * (s ∪ t) ⊆ s * t :=
  image2_inter_union_subset mul_comm


@[to_additive]
theorem union_mul_inter_subset : (s ∪ t) * (s ∩ t) ⊆ s * t :=
  image2_union_inter_subset mul_comm


/-- `Set α` is a `MulOneClass` under pointwise operations if `α` is. -/
@[to_additive "`Set α` is an `AddZeroClass` under pointwise operations if `α` is."]
protected noncomputable def mulOneClass : MulOneClass (Set α) :=
  { Set.one, Set.mul with
    mul_one := image2_right_identity mul_one
    one_mul := image2_left_identity one_mul }


@[to_additive]
theorem subset_mul_left (s : Set α) {t : Set α} (ht : (1 : α) ∈ t) : s ⊆ s * t := fun x hx =>
  ⟨x, hx, 1, ht, mul_one _⟩


@[to_additive]
theorem subset_mul_right {s : Set α} (t : Set α) (hs : (1 : α) ∈ s) : t ⊆ s * t := fun x hx =>
  ⟨1, hs, x, hx, one_mul _⟩


/-- The singleton operation as a `MonoidHom`. -/
@[to_additive "The singleton operation as an `AddMonoidHom`."]
noncomputable def singletonMonoidHom : α →* Set α :=
  { singletonMulHom, singletonOneHom with }


@[to_additive (attr := simp)]
theorem coe_singletonMonoidHom : (singletonMonoidHom : α → Set α) = singleton :=
  rfl


@[to_additive (attr := simp)]
theorem singletonMonoidHom_apply (a : α) : singletonMonoidHom a = {a} :=
  rfl


/-- `Set α` is a `Monoid` under pointwise operations if `α` is. -/
@[to_additive "`Set α` is an `AddMonoid` under pointwise operations if `α` is."]
protected noncomputable def monoid : Monoid (Set α) :=
  { Set.semigroup, Set.mulOneClass, @Set.NPow α _ _ with }


@[to_additive]
protected lemma pow_right_monotone (hs : 1 ∈ s) : Monotone (s ^ ·) :=
  pow_right_monotone <| one_subset.2 hs


@[to_additive (attr := gcongr)]
lemma pow_subset_pow_left (hst : s ⊆ t) : s ^ n ⊆ t ^ n := pow_left_mono _ hst


@[to_additive (attr := gcongr)]
lemma pow_subset_pow_right (hs : 1 ∈ s) (hmn : m ≤ n) : s ^ m ⊆ s ^ n :=
  Set.pow_right_monotone hs hmn


@[to_additive (attr := gcongr)]
lemma pow_subset_pow (hst : s ⊆ t) (ht : 1 ∈ t) (hmn : m ≤ n) : s ^ m ⊆ t ^ n :=
  (pow_subset_pow_left hst).trans (pow_subset_pow_right ht hmn)


@[to_additive]
lemma subset_pow (hs : 1 ∈ s) (hn : n ≠ 0) : s ⊆ s ^ n := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    s : Set α
    n : Nat
    hs : Membership.mem s 1
    hn : Ne n 0
    ⊢ HasSubset.Subset s (HPow.hPow s n)
  -/
  simpa using pow_subset_pow_right hs <| Nat.one_le_iff_ne_zero.2 hn
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-19")] alias pow_subset_pow_of_one_mem := pow_subset_pow_right


@[deprecated (since := "2024-11-19")]
alias nsmul_subset_nsmul_of_zero_mem := nsmul_subset_nsmul_right


@[to_additive]
lemma pow_subset_pow_mul_of_sq_subset_mul (hst : s ^ 2 ⊆ t * s) (hn : n ≠ 0) :
    s ^ n ⊆ t ^ (n - 1) * s := pow_le_pow_mul_of_sq_le_mul hst hn


@[to_additive (attr := simp) nsmul_empty]
                                                                                 /-
                                                                                   α : Type u_2
                                                                                   inst✝ : Monoid α
                                                                                   n✝ n : Nat
                                                                                   hn : Ne (HAdd.hAdd n 1) 0
                                                                                   ⊢ Eq (HPow.hPow EmptyCollection.emptyCollection (HAdd.hAdd n 1)) EmptyCollecti …
                                                                                 -/
lemma empty_pow (hn : n ≠ 0) : (∅ : Set α) ^ n = ∅ := match n with | n + 1 => by simp [pow_succ]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[deprecated (since := "2024-10-21")] alias empty_nsmul := nsmul_empty


@[to_additive]
lemma Nonempty.pow (hs : s.Nonempty) : ∀ {n}, (s ^ n).Nonempty
            /-
              α : Type u_2
              inst✝ : Monoid α
              s : Set α
              hs : s.Nonempty
              ⊢ (HPow.hPow s 0).Nonempty
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_2
                  inst✝ : Monoid α
                  s : Set α
                  hs : s.Nonempty
                  n : Nat
                  ⊢ (HPow.hPow s (HAdd.hAdd n 1)).Nonempty
                -/
  | n + 1 => by rw [pow_succ]; exact hs.pow.mul hs
                               /-
                                 🎉 no goals
                               -/


set_option push_neg.use_distrib true in
@[to_additive (attr := simp)] lemma pow_eq_empty : s ^ n = ∅ ↔ s = ∅ ∧ n ≠ 0 := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    s : Set α
    n : Nat
    ⊢ Iff (Eq (HPow.hPow s n) EmptyCollection.emptyCollection) (And (Eq s EmptyCol …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : Monoid α
      s : Set α
      n : Nat
      ⊢ Eq (HPow.hPow s n) EmptyCollection.emptyCollection → And (Eq s EmptyCollecti …
    -/
  · contrapose!
    /-
      case mp
      α : Type u_2
      inst✝ : Monoid α
      s : Set α
      n : Nat
      ⊢ Or s.Nonempty (Eq n 0) → (HPow.hPow s n).Nonempty
    -/
    rintro (hs | rfl)
      /-
        case mp.inl
        α : Type u_2
        inst✝ : Monoid α
        s : Set α
        n : Nat
        hs : s.Nonempty
        ⊢ (HPow.hPow s n).Nonempty
      -/
    · exact hs.pow
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        α : Type u_2
        inst✝ : Monoid α
        s : Set α
        ⊢ (HPow.hPow s 0).Nonempty
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_2
      inst✝ : Monoid α
      s : Set α
      n : Nat
      ⊢ And (Eq s EmptyCollection.emptyCollection) (Ne n 0) → Eq (HPow.hPow s n) Emp …
    -/
  · rintro ⟨rfl, hn⟩
    /-
      case mpr.intro
      α : Type u_2
      inst✝ : Monoid α
      n : Nat
      hn : Ne n 0
      ⊢ Eq (HPow.hPow EmptyCollection.emptyCollection n) EmptyCollection.emptyCollec …
    -/
    exact empty_pow hn
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp) nsmul_singleton]
lemma singleton_pow (a : α) : ∀ n, ({a} : Set α) ^ n = {a ^ n}
            /-
              α : Type u_2
              inst✝ : Monoid α
              a : α
              ⊢ Eq (HPow.hPow (Singleton.singleton a) 0) (Singleton.singleton (HPow.hPow a 0))
            -/
  | 0 => by simp [singleton_one]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_2
                  inst✝ : Monoid α
                  a : α
                  n : Nat
                  ⊢ Eq (HPow.hPow (Singleton.singleton a) (HAdd.hAdd n 1)) (Singleton.singleton  …
                -/
  | n + 1 => by simp [pow_succ, singleton_pow _ n]
                /-
                  🎉 no goals
                -/


@[to_additive] lemma pow_mem_pow (ha : a ∈ s) : a ^ n ∈ s ^ n := by
  /-
    α : Type u_2
    inst✝ : Monoid α
    s : Set α
    a : α
    n : Nat
    ha : Membership.mem s a
    ⊢ Membership.mem (HPow.hPow s n) (HPow.hPow a n)
  -/
  simpa using pow_subset_pow_left (singleton_subset_iff.2 ha)
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  α : Type u_2
                                                                  inst✝ : Monoid α
                                                                  s : Set α
                                                                  n : Nat
                                                                  hs : Membership.mem s 1
                                                                  ⊢ Membership.mem (HPow.hPow s n) 1
                                                                -/
@[to_additive] lemma one_mem_pow (hs : 1 ∈ s) : 1 ∈ s ^ n := by simpa using pow_mem_pow hs
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive]
                                                           /-
                                                             α : Type u_2
                                                             inst✝ : Monoid α
                                                             s t : Set α
                                                             n : Nat
                                                             ⊢ HasSubset.Subset (HPow.hPow (Inter.inter s t) n) (Inter.inter (HPow.hPow s n …
                                                           -/
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
lemma inter_pow_subset : (s ∩ t) ^ n ⊆ s ^ n ∩ t ^ n := by apply subset_inter <;> gcongr <;> simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[to_additive]
theorem mul_univ_of_one_mem (hs : (1 : α) ∈ s) : s * univ = univ :=
  eq_univ_iff_forall.2 fun _ => mem_mul.2 ⟨_, hs, _, mem_univ _, one_mul _⟩


@[to_additive]
theorem univ_mul_of_one_mem (ht : (1 : α) ∈ t) : univ * t = univ :=
  eq_univ_iff_forall.2 fun _ => mem_mul.2 ⟨_, mem_univ _, _, ht, mul_one _⟩


@[to_additive (attr := simp)]
theorem univ_mul_univ : (univ : Set α) * univ = univ :=
  mul_univ_of_one_mem <| mem_univ _


@[to_additive (attr := simp) nsmul_univ]
theorem univ_pow : ∀ {n : ℕ}, n ≠ 0 → (univ : Set α) ^ n = univ
  | 0 => fun h => (h rfl).elim
  | 1 => fun _ => pow_one _
                         /-
                           α : Type u_2
                           inst✝ : Monoid α
                           n : Nat
                           x✝ : Ne (HAdd.hAdd n 2) 0
                           ⊢ Eq (HPow.hPow Set.univ (HAdd.hAdd n 2)) Set.univ
                         -/
  | n + 2 => fun _ => by rw [pow_succ, univ_pow n.succ_ne_zero, univ_mul_univ]
                         /-
                           🎉 no goals
                         -/


@[to_additive]
protected theorem _root_.IsUnit.set : IsUnit a → IsUnit ({a} : Set α) :=
  IsUnit.map (singletonMonoidHom : α →* Set α)


@[to_additive]
lemma Nontrivial.mul_left : t.Nontrivial → s.Nonempty → (s * t).Nontrivial := by
  /-
    α : Type u_2
    inst✝¹ : Mul α
    inst✝ : IsLeftCancelMul α
    s t : Set α
    ⊢ t.Nontrivial → s.Nonempty → (HMul.hMul s t).Nontrivial
  -/
  rintro ⟨a, ha, b, hb, hab⟩ ⟨c, hc⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : Mul α
    inst✝ : IsLeftCancelMul α
    s t : Set α
    a : α
    ha : Membership.mem t a
    b : α
    hb : Membership.mem t b
    hab : Ne a b
    c : α
    hc : Membership.mem s c
    ⊢ (HMul.hMul s t).Nontrivial
  -/
  exact ⟨c * a, mul_mem_mul hc ha, c * b, mul_mem_mul hc hb, by simpa⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Nontrivial.mul (hs : s.Nontrivial) (ht : t.Nontrivial) : (s * t).Nontrivial :=
  ht.mul_left hs.nonempty


@[to_additive]
lemma Nontrivial.mul_right : s.Nontrivial → t.Nonempty → (s * t).Nontrivial := by
  /-
    α : Type u_2
    inst✝¹ : Mul α
    inst✝ : IsRightCancelMul α
    s t : Set α
    ⊢ s.Nontrivial → t.Nonempty → (HMul.hMul s t).Nontrivial
  -/
  rintro ⟨a, ha, b, hb, hab⟩ ⟨c, hc⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : Mul α
    inst✝ : IsRightCancelMul α
    s t : Set α
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Ne a b
    c : α
    hc : Membership.mem t c
    ⊢ (HMul.hMul s t).Nontrivial
  -/
  exact ⟨a * c, mul_mem_mul ha hc, b * c, mul_mem_mul hb hc, by simpa⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Nontrivial.pow (hs : s.Nontrivial) : ∀ {n}, n ≠ 0 → (s ^ n).Nontrivial
               /-
                 α : Type u_2
                 inst✝ : CancelMonoid α
                 s : Set α
                 hs : s.Nontrivial
                 x✝ : Ne 1 0
                 ⊢ (HPow.hPow s 1).Nontrivial
               -/
  | 1, _ => by simpa
               /-
                 🎉 no goals
               -/
                   /-
                     α : Type u_2
                     inst✝ : CancelMonoid α
                     s : Set α
                     hs : s.Nontrivial
                     n : Nat
                     x✝ : Ne (HAdd.hAdd n 2) 0
                     ⊢ (HPow.hPow s (HAdd.hAdd n 2)).Nontrivial
                   -/
  | n + 2, _ => by simpa [pow_succ] using (hs.pow n.succ_ne_zero).mul hs
                   /-
                     🎉 no goals
                   -/


/-- `Set α` is a `CommMonoid` under pointwise operations if `α` is. -/
@[to_additive "`Set α` is an `AddCommMonoid` under pointwise operations if `α` is."]
protected noncomputable def commMonoid [CommMonoid α] : CommMonoid (Set α) :=
  { Set.monoid, Set.commSemigroup with }


@[to_additive]
protected theorem mul_eq_one_iff : s * t = 1 ↔ ∃ a b, s = {a} ∧ t = {b} ∧ a * b = 1 := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    s t : Set α
    ⊢ Iff (Eq (HMul.hMul s t) 1) (Exists fun a => Exists fun b => And (Eq s (Singl …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : DivisionMonoid α
      s t : Set α
      h : Eq (HMul.hMul s t) 1
      ⊢ Exists fun a => Exists fun b => And (Eq s (Singleton.singleton a)) (And (Eq  …
    -/
  · have hst : (s * t).Nonempty := h.symm.subst one_nonempty
    /-
      case refine_1
      α : Type u_2
      inst✝ : DivisionMonoid α
      s t : Set α
      h : Eq (HMul.hMul s t) 1
      hst : (HMul.hMul s t).Nonempty
      ⊢ Exists fun a => Exists fun b => And (Eq s (Singleton.singleton a)) (And (Eq  …
    -/
    obtain ⟨a, ha⟩ := hst.of_image2_left
    /-
      case refine_1.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      s t : Set α
      h : Eq (HMul.hMul s t) 1
      hst : (HMul.hMul s t).Nonempty
      a : α
      ha : Membership.mem s a
      ⊢ Exists fun a => Exists fun b => And (Eq s (Singleton.singleton a)) (And (Eq  …
    -/
    obtain ⟨b, hb⟩ := hst.of_image2_right
    have H : ∀ {a b}, a ∈ s → b ∈ t → a * b = (1 : α) := fun {a b} ha hb =>
      h.subset <| mem_image2_of_mem ha hb
    /-
      case refine_1.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      s t : Set α
      h : Eq (HMul.hMul s t) 1
      hst : (HMul.hMul s t).Nonempty
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem t b
      H : ∀ {a b : α}, Membership.mem s a → Membership.mem t b → Eq (HMul.hMul a b) 1
      ⊢ Exists fun a => Exists fun b => And (Eq s (Singleton.singleton a)) (And (Eq  …
    -/
    refine ⟨a, b, ?_, ?_, H ha hb⟩ <;> refine eq_singleton_iff_unique_mem.2 ⟨‹_›, fun x hx => ?_⟩
      /-
        case refine_1.intro.intro.refine_1
        α : Type u_2
        inst✝ : DivisionMonoid α
        s t : Set α
        h : Eq (HMul.hMul s t) 1
        hst : (HMul.hMul s t).Nonempty
        a : α
        ha : Membership.mem s a
        b : α
        hb : Membership.mem t b
        H : ∀ {a b : α}, Membership.mem s a → Membership.mem t b → Eq (HMul.hMul a b) 1
        x : α
        hx : Membership.mem s x
        ⊢ Eq x a
      -/
    · exact (eq_inv_of_mul_eq_one_left <| H hx hb).trans (inv_eq_of_mul_eq_one_left <| H ha hb)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.refine_2
        α : Type u_2
        inst✝ : DivisionMonoid α
        s t : Set α
        h : Eq (HMul.hMul s t) 1
        hst : (HMul.hMul s t).Nonempty
        a : α
        ha : Membership.mem s a
        b : α
        hb : Membership.mem t b
        H : ∀ {a b : α}, Membership.mem s a → Membership.mem t b → Eq (HMul.hMul a b) 1
        x : α
        hx : Membership.mem t x
        ⊢ Eq x b
      -/
    · exact (eq_inv_of_mul_eq_one_right <| H ha hx).trans (inv_eq_of_mul_eq_one_right <| H ha hb)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : DivisionMonoid α
      s t : Set α
      ⊢ (Exists fun a => Exists fun b => And (Eq s (Singleton.singleton a)) (And (Eq …
    -/
  · rintro ⟨b, c, rfl, rfl, h⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      b c : α
      h : Eq (HMul.hMul b c) 1
      ⊢ Eq (HMul.hMul (Singleton.singleton b) (Singleton.singleton c)) 1
    -/
    rw [singleton_mul_singleton, h, singleton_one]
    /-
      🎉 no goals
    -/


/-- `Set α` is a division monoid under pointwise operations if `α` is. -/
@[to_additive
    "`Set α` is a subtraction monoid under pointwise operations if `α` is."]
protected noncomputable def divisionMonoid : DivisionMonoid (Set α) :=
  { Set.monoid, Set.involutiveInv, Set.div, @Set.ZPow α _ _ _ with
    mul_inv_rev := fun s t => by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : DivisionMonoid α
        s✝ t✝ : Set α
        n : Int
        s t : Set α
        ⊢ Eq (Inv.inv (HMul.hMul s t)) (HMul.hMul (Inv.inv t) (Inv.inv s))
      -/
      simp_rw [← image_inv_eq_inv]
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : DivisionMonoid α
        s✝ t✝ : Set α
        n : Int
        s t : Set α
        ⊢ Eq (Set.image (fun x => Inv.inv x) (HMul.hMul s t)) (HMul.hMul (Set.image (f …
      -/
      exact image_image2_antidistrib mul_inv_rev
      /-
        🎉 no goals
      -/
    inv_eq_of_mul := fun s t h => by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : DivisionMonoid α
        s✝ t✝ : Set α
        n : Int
        s t : Set α
        ⊢ Eq (HDiv.hDiv s t) (HMul.hMul s (Inv.inv t))
      -/
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : DivisionMonoid α
        s✝ t✝ : Set α
        n : Int
        s t : Set α
        h : Eq (HMul.hMul s t) 1
        ⊢ Eq (Inv.inv s) t
      -/
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : DivisionMonoid α
        s✝ t✝ : Set α
        n : Int
        s t : Set α
        ⊢ Eq (Set.image id (HDiv.hDiv s t)) (HMul.hMul s (Set.image (fun x => Inv.inv  …
      -/
      obtain ⟨a, b, rfl, rfl, hab⟩ := Set.mul_eq_one_iff.1 h
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : DivisionMonoid α
        s t : Set α
        n : Int
        a b : α
        hab : Eq (HMul.hMul a b) 1
        h : Eq (HMul.hMul (Singleton.singleton a) (Singleton.singleton b)) 1
        ⊢ Eq (Inv.inv (Singleton.singleton a)) (Singleton.singleton b)
      -/
      rw [inv_singleton, inv_eq_of_mul_eq_one_right hab]
      /-
        🎉 no goals
      -/
    div_eq_mul_inv := fun s t => by
      rw [← image_id (s / t), ← image_inv_eq_inv]
      exact image_image2_distrib_right div_eq_mul_inv }


@[to_additive (attr := simp 500)]
theorem isUnit_iff : IsUnit s ↔ ∃ a, s = {a} ∧ IsUnit a := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    s : Set α
    ⊢ Iff (IsUnit s) (Exists fun a => And (Eq s (Singleton.singleton a)) (IsUnit a))
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : DivisionMonoid α
      s : Set α
      ⊢ IsUnit s → Exists fun a => And (Eq s (Singleton.singleton a)) (IsUnit a)
    -/
  · rintro ⟨u, rfl⟩
    /-
      case mp.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Set α)
      ⊢ Exists fun a => And (Eq (↑u) (Singleton.singleton a)) (IsUnit a)
    -/
    obtain ⟨a, b, ha, hb, h⟩ := Set.mul_eq_one_iff.1 u.mul_inv
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Set α)
      a b : α
      ha : Eq (↑u) (Singleton.singleton a)
      hb : Eq (↑(Inv.inv u)) (Singleton.singleton b)
      h : Eq (HMul.hMul a b) 1
      ⊢ Exists fun a => And (Eq (↑u) (Singleton.singleton a)) (IsUnit a)
    -/
    refine ⟨a, ha, ⟨a, b, h, singleton_injective ?_⟩, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Set α)
      a b : α
      ha : Eq (↑u) (Singleton.singleton a)
      hb : Eq (↑(Inv.inv u)) (Singleton.singleton b)
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (Singleton.singleton (HMul.hMul b a)) (Singleton.singleton 1)
    -/
    rw [← singleton_mul_singleton, ← ha, ← hb]
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      u : Units (Set α)
      a b : α
      ha : Eq (↑u) (Singleton.singleton a)
      hb : Eq (↑(Inv.inv u)) (Singleton.singleton b)
      h : Eq (HMul.hMul a b) 1
      ⊢ Eq (HMul.hMul ↑(Inv.inv u) ↑u) (Singleton.singleton 1)
    -/
    exact u.inv_mul
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : DivisionMonoid α
      s : Set α
      ⊢ (Exists fun a => And (Eq s (Singleton.singleton a)) (IsUnit a)) → IsUnit s
    -/
  · rintro ⟨a, rfl, ha⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      a : α
      ha : IsUnit a
      ⊢ IsUnit (Singleton.singleton a)
    -/
    exact ha.set
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
                                                         /-
                                                           α : Type u_2
                                                           inst✝ : DivisionMonoid α
                                                           ⊢ Eq (HDiv.hDiv Set.univ Set.univ) Set.univ
                                                         -/
lemma univ_div_univ : (univ / univ : Set α) = univ := by simp [div_eq_mul_inv]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive] lemma subset_div_left (ht : 1 ∈ t) : s ⊆ s / t := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    s t : Set α
    ht : Membership.mem t 1
    ⊢ HasSubset.Subset s (HDiv.hDiv s t)
  -/
  rw [div_eq_mul_inv]; exact subset_mul_left _ <| by simpa
                       /-
                         🎉 no goals
                       -/


@[to_additive] lemma inv_subset_div_right (hs : 1 ∈ s) : t⁻¹ ⊆ s / t := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    s t : Set α
    hs : Membership.mem s 1
    ⊢ HasSubset.Subset (Inv.inv t) (HDiv.hDiv s t)
  -/
  rw [div_eq_mul_inv]; exact subset_mul_right _ hs
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp) zsmul_empty]
                                                          /-
                                                            α : Type u_2
                                                            inst✝ : DivisionMonoid α
                                                            n : Int
                                                            hn : Ne n 0
                                                            ⊢ Eq (HPow.hPow EmptyCollection.emptyCollection n) EmptyCollection.emptyCollec …
                                                          -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
lemma empty_zpow (hn : n ≠ 0) : (∅ : Set α) ^ n = ∅ := by cases n <;> aesop
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive]
lemma Nonempty.zpow (hs : s.Nonempty) : ∀ {n : ℤ}, (s ^ n).Nonempty
  | (n : ℕ) => hs.pow
                     /-
                       α : Type u_2
                       inst✝ : DivisionMonoid α
                       s : Set α
                       hs : s.Nonempty
                       n : Nat
                       ⊢ (HPow.hPow s (Int.negSucc n)).Nonempty
                     -/
  | .negSucc n => by simpa using hs.pow
                     /-
                       🎉 no goals
                     -/


set_option push_neg.use_distrib true in
@[to_additive (attr := simp)] lemma zpow_eq_empty : s ^ n = ∅ ↔ s = ∅ ∧ n ≠ 0 := by
  /-
    α : Type u_2
    inst✝ : DivisionMonoid α
    s : Set α
    n : Int
    ⊢ Iff (Eq (HPow.hPow s n) EmptyCollection.emptyCollection) (And (Eq s EmptyCol …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : DivisionMonoid α
      s : Set α
      n : Int
      ⊢ Eq (HPow.hPow s n) EmptyCollection.emptyCollection → And (Eq s EmptyCollecti …
    -/
  · contrapose!
    /-
      case mp
      α : Type u_2
      inst✝ : DivisionMonoid α
      s : Set α
      n : Int
      ⊢ Or s.Nonempty (Eq n 0) → (HPow.hPow s n).Nonempty
    -/
    rintro (hs | rfl)
      /-
        case mp.inl
        α : Type u_2
        inst✝ : DivisionMonoid α
        s : Set α
        n : Int
        hs : s.Nonempty
        ⊢ (HPow.hPow s n).Nonempty
      -/
    · exact hs.zpow
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        α : Type u_2
        inst✝ : DivisionMonoid α
        s : Set α
        ⊢ (HPow.hPow s 0).Nonempty
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_2
      inst✝ : DivisionMonoid α
      s : Set α
      n : Int
      ⊢ And (Eq s EmptyCollection.emptyCollection) (Ne n 0) → Eq (HPow.hPow s n) Emp …
    -/
  · rintro ⟨rfl, hn⟩
    /-
      case mpr.intro
      α : Type u_2
      inst✝ : DivisionMonoid α
      n : Int
      hn : Ne n 0
      ⊢ Eq (HPow.hPow EmptyCollection.emptyCollection n) EmptyCollection.emptyCollec …
    -/
    exact empty_zpow hn
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp) zsmul_singleton]
                                                                         /-
                                                                           α : Type u_2
                                                                           inst✝ : DivisionMonoid α
                                                                           a : α
                                                                           n : Int
                                                                           ⊢ Eq (HPow.hPow (Singleton.singleton a) n) (Singleton.singleton (HPow.hPow a n))
                                                                         -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
lemma singleton_zpow (a : α) (n : ℤ) : ({a} : Set α) ^ n = {a ^ n} := by cases n <;> simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- `Set α` is a commutative division monoid under pointwise operations if `α` is. -/
@[to_additive subtractionCommMonoid
      "`Set α` is a commutative subtraction monoid under pointwise operations if `α` is."]
protected noncomputable def divisionCommMonoid [DivisionCommMonoid α] :
    DivisionCommMonoid (Set α) :=
  { Set.divisionMonoid, Set.commSemigroup with }


@[to_additive (attr := simp)]
theorem one_mem_div_iff : (1 : α) ∈ s / t ↔ ¬Disjoint s t := by
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    ⊢ Iff (Membership.mem (HDiv.hDiv s t) 1) (Not (Disjoint s t))
  -/
  simp [not_disjoint_iff_nonempty_inter, mem_div, div_eq_one, Set.Nonempty]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma one_mem_inv_mul_iff : (1 : α) ∈ t⁻¹ * s ↔ ¬Disjoint s t := by
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    ⊢ Iff (Membership.mem (HMul.hMul (Inv.inv t) s) 1) (Not (Disjoint s t))
  -/
  aesop (add simp [not_disjoint_iff_nonempty_inter, mem_mul, mul_eq_one_iff_eq_inv, Set.Nonempty])
  /-
    🎉 no goals
  -/


@[to_additive]
theorem not_one_mem_div_iff : (1 : α) ∉ s / t ↔ Disjoint s t :=
  one_mem_div_iff.not_left


@[to_additive]
lemma not_one_mem_inv_mul_iff : (1 : α) ∉ t⁻¹ * s ↔ Disjoint s t := one_mem_inv_mul_iff.not_left


alias ⟨_, _root_.Disjoint.one_not_mem_div_set⟩ := not_one_mem_div_iff


attribute [to_additive] Disjoint.one_not_mem_div_set


@[to_additive]
theorem Nonempty.one_mem_div (h : s.Nonempty) : (1 : α) ∈ s / s :=
  let ⟨a, ha⟩ := h
  mem_div.2 ⟨a, ha, a, ha, div_self' _⟩


@[to_additive]
theorem isUnit_singleton (a : α) : IsUnit ({a} : Set α) :=
  (Group.isUnit a).set


@[to_additive (attr := simp)]
theorem isUnit_iff_singleton : IsUnit s ↔ ∃ a, s = {a} := by
  /-
    α : Type u_2
    inst✝ : Group α
    s : Set α
    ⊢ Iff (IsUnit s) (Exists fun a => Eq s (Singleton.singleton a))
  -/
  simp only [isUnit_iff, Group.isUnit, and_true]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem image_mul_left : (a * ·) '' t = (a⁻¹ * ·) ⁻¹' t := by
  /-
    α : Type u_2
    inst✝ : Group α
    t : Set α
    a : α
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) t) (Set.preimage (fun x => HMul.hMul  …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  rw [image_eq_preimage_of_inverse] <;> intro c <;> simp
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive (attr := simp)]
theorem image_mul_right : (· * b) '' t = (· * b⁻¹) ⁻¹' t := by
  /-
    α : Type u_2
    inst✝ : Group α
    t : Set α
    b : α
    ⊢ Eq (Set.image (fun x => HMul.hMul x b) t) (Set.preimage (fun x => HMul.hMul  …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  rw [image_eq_preimage_of_inverse] <;> intro c <;> simp
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive]
                                                               /-
                                                                 α : Type u_2
                                                                 inst✝ : Group α
                                                                 t : Set α
                                                                 a : α
                                                                 ⊢ Eq (Set.image (fun x => HMul.hMul (Inv.inv a) x) t) (Set.preimage (fun x =>  …
                                                               -/
theorem image_mul_left' : (a⁻¹ * ·) '' t = (a * ·) ⁻¹' t := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive]
                                                                /-
                                                                  α : Type u_2
                                                                  inst✝ : Group α
                                                                  t : Set α
                                                                  b : α
                                                                  ⊢ Eq (Set.image (fun x => HMul.hMul x (Inv.inv b)) t) (Set.preimage (fun x =>  …
                                                                -/
theorem image_mul_right' : (· * b⁻¹) '' t = (· * b) ⁻¹' t := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive (attr := simp)]
theorem preimage_mul_left_singleton : (a * ·) ⁻¹' {b} = {a⁻¹ * b} := by
  /-
    α : Type u_2
    inst✝ : Group α
    a b : α
    ⊢ Eq (Set.preimage (fun x => HMul.hMul a x) (Singleton.singleton b)) (Singleto …
  -/
  rw [← image_mul_left', image_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem preimage_mul_right_singleton : (· * a) ⁻¹' {b} = {b * a⁻¹} := by
  /-
    α : Type u_2
    inst✝ : Group α
    a b : α
    ⊢ Eq (Set.preimage (fun x => HMul.hMul x a) (Singleton.singleton b)) (Singleto …
  -/
  rw [← image_mul_right', image_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem preimage_mul_left_one : (a * ·) ⁻¹' 1 = {a⁻¹} := by
  /-
    α : Type u_2
    inst✝ : Group α
    a : α
    ⊢ Eq (Set.preimage (fun x => HMul.hMul a x) 1) (Singleton.singleton (Inv.inv a))
  -/
  rw [← image_mul_left', image_one, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem preimage_mul_right_one : (· * b) ⁻¹' 1 = {b⁻¹} := by
  /-
    α : Type u_2
    inst✝ : Group α
    b : α
    ⊢ Eq (Set.preimage (fun x => HMul.hMul x b) 1) (Singleton.singleton (Inv.inv b))
  -/
  rw [← image_mul_right', image_one, one_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                             /-
                                                               α : Type u_2
                                                               inst✝ : Group α
                                                               a : α
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul (Inv.inv a) x) 1) (Singleton.singleton a)
                                                             -/
theorem preimage_mul_left_one' : (a⁻¹ * ·) ⁻¹' 1 = {a} := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
                                                              /-
                                                                α : Type u_2
                                                                inst✝ : Group α
                                                                b : α
                                                                ⊢ Eq (Set.preimage (fun x => HMul.hMul x (Inv.inv b)) 1) (Singleton.singleton b)
                                                              -/
theorem preimage_mul_right_one' : (· * b⁻¹) ⁻¹' 1 = {b} := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[to_additive (attr := simp)]
theorem mul_univ (hs : s.Nonempty) : s * (univ : Set α) = univ :=
  let ⟨a, ha⟩ := hs
  eq_univ_of_forall fun b => ⟨a, ha, a⁻¹ * b, trivial, mul_inv_cancel_left ..⟩


@[to_additive (attr := simp)]
theorem univ_mul (ht : t.Nonempty) : (univ : Set α) * t = univ :=
  let ⟨a, ha⟩ := ht
  eq_univ_of_forall fun b => ⟨b * a⁻¹, trivial, a, ha, inv_mul_cancel_right ..⟩


@[to_additive]
lemma image_inv [DivisionMonoid β] [FunLike F α β] [MonoidHomClass F α β] (f : F) (s : Set α) :
    f '' s⁻¹ = (f '' s)⁻¹ := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    f : F
    s : Set α
    ⊢ Eq (Set.image (⇑f) (Inv.inv s)) (Inv.inv (Set.image (⇑f) s))
  -/
  rw [← image_inv_eq_inv, ← image_inv_eq_inv]; exact image_comm (map_inv _)
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem image_mul : m '' (s * t) = m '' s * m '' t :=
  image_image2_distrib <| map_mul m


@[to_additive]
lemma mul_subset_range {s t : Set β} (hs : s ⊆ range m) (ht : t ⊆ range m) : s * t ⊆ range m := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Mul α
    inst✝² : Mul β
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    ⊢ HasSubset.Subset (HMul.hMul s t) (Set.range ⇑m)
  -/
  rintro _ ⟨a, ha, b, hb, rfl⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Mul α
    inst✝² : Mul β
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    a : β
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    ⊢ Membership.mem (Set.range ⇑m) ((fun x1 x2 => HMul.hMul x1 x2) a b)
  -/
  obtain ⟨a, rfl⟩ := hs ha
  /-
    case intro.intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Mul α
    inst✝² : Mul β
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    b : β
    hb : Membership.mem t b
    a : α
    ha : Membership.mem s (m a)
    ⊢ Membership.mem (Set.range ⇑m) ((fun x1 x2 => HMul.hMul x1 x2) (m a) b)
  -/
  obtain ⟨b, rfl⟩ := ht hb
  /-
    case intro.intro.intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Mul α
    inst✝² : Mul β
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    a : α
    ha : Membership.mem s (m a)
    b : α
    hb : Membership.mem t (m b)
    ⊢ Membership.mem (Set.range ⇑m) ((fun x1 x2 => HMul.hMul x1 x2) (m a) (m b))
  -/
  exact ⟨a * b, map_mul ..⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem preimage_mul_preimage_subset {s t : Set β} : m ⁻¹' s * m ⁻¹' t ⊆ m ⁻¹' (s * t) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Mul α
    inst✝² : Mul β
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    m : F
    s t : Set β
    ⊢ HasSubset.Subset (HMul.hMul (Set.preimage (⇑m) s) (Set.preimage (⇑m) t)) (Se …
  -/
  rintro _ ⟨_, _, _, _, rfl⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Mul α
    inst✝² : Mul β
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    m : F
    s t : Set β
    w✝¹ : α
    left✝¹ : Membership.mem (Set.preimage (⇑m) s) w✝¹
    w✝ : α
    left✝ : Membership.mem (Set.preimage (⇑m) t) w✝
    ⊢ Membership.mem (Set.preimage (⇑m) (HMul.hMul s t)) ((fun x1 x2 => HMul.hMul  …
  -/
  exact ⟨_, ‹_›, _, ‹_›, (map_mul m ..).symm⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma preimage_mul (hm : Injective m) {s t : Set β} (hs : s ⊆ range m) (ht : t ⊆ range m) :
    m ⁻¹' (s * t) = m ⁻¹' s * m ⁻¹' t :=
  hm.image_injective <| by
    rw [image_mul, image_preimage_eq_iff.2 hs, image_preimage_eq_iff.2 ht,
      image_preimage_eq_iff.2 (mul_subset_range m hs ht)]


@[to_additive]
lemma image_pow_of_ne_zero [MulHomClass F α β] :
    ∀ {n}, n ≠ 0 → ∀ (f : F) (s : Set α), f '' (s ^ n) = (f '' s) ^ n
               /-
                 F : Type u_1
                 α : Type u_2
                 β : Type u_3
                 inst✝³ : Monoid α
                 inst✝² : Monoid β
                 inst✝¹ : FunLike F α β
                 inst✝ : MulHomClass F α β
                 x✝ : Ne 1 0
                 ⊢ ∀ (f : F) (s : Set α), Eq (Set.image (⇑f) (HPow.hPow s 1)) (HPow.hPow (Set.i …
               -/
  | 1, _ => by simp
               /-
                 🎉 no goals
               -/
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     inst✝³ : Monoid α
                     inst✝² : Monoid β
                     inst✝¹ : FunLike F α β
                     inst✝ : MulHomClass F α β
                     n : Nat
                     x✝ : Ne (HAdd.hAdd n 2) 0
                     ⊢ ∀ (f : F) (s : Set α), Eq (Set.image (⇑f) (HPow.hPow s (HAdd.hAdd n 2))) (HP …
                   -/
  | n + 2, _ => by simp [image_mul, pow_succ _ n.succ, image_pow_of_ne_zero]
                   /-
                     🎉 no goals
                   -/


@[to_additive]
lemma image_pow [MonoidHomClass F α β] (f : F) (s : Set α) : ∀ n, f '' (s ^ n) = (f '' s) ^ n
            /-
              F : Type u_1
              α : Type u_2
              β : Type u_3
              inst✝³ : Monoid α
              inst✝² : Monoid β
              inst✝¹ : FunLike F α β
              inst✝ : MonoidHomClass F α β
              f : F
              s : Set α
              ⊢ Eq (Set.image (⇑f) (HPow.hPow s 0)) (HPow.hPow (Set.image (⇑f) s) 0)
            -/
  | 0 => by simp [singleton_one]
            /-
              🎉 no goals
            -/
  | n + 1 => image_pow_of_ne_zero n.succ_ne_zero ..


@[to_additive]
theorem image_div : m '' (s / t) = m '' s / m '' t :=
  image_image2_distrib <| map_div m


@[to_additive]
lemma div_subset_range {s t : Set β} (hs : s ⊆ range m) (ht : t ⊆ range m) : s / t ⊆ range m := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    ⊢ HasSubset.Subset (HDiv.hDiv s t) (Set.range ⇑m)
  -/
  rintro _ ⟨a, ha, b, hb, rfl⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    a : β
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    ⊢ Membership.mem (Set.range ⇑m) ((fun x1 x2 => HDiv.hDiv x1 x2) a b)
  -/
  obtain ⟨a, rfl⟩ := hs ha
  /-
    case intro.intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    b : β
    hb : Membership.mem t b
    a : α
    ha : Membership.mem s (m a)
    ⊢ Membership.mem (Set.range ⇑m) ((fun x1 x2 => HDiv.hDiv x1 x2) (m a) b)
  -/
  obtain ⟨b, rfl⟩ := ht hb
  /-
    case intro.intro.intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : F
    s t : Set β
    hs : HasSubset.Subset s (Set.range ⇑m)
    ht : HasSubset.Subset t (Set.range ⇑m)
    a : α
    ha : Membership.mem s (m a)
    b : α
    hb : Membership.mem t (m b)
    ⊢ Membership.mem (Set.range ⇑m) ((fun x1 x2 => HDiv.hDiv x1 x2) (m a) (m b))
  -/
  exact ⟨a / b, map_div ..⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem preimage_div_preimage_subset {s t : Set β} : m ⁻¹' s / m ⁻¹' t ⊆ m ⁻¹' (s / t) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : F
    s t : Set β
    ⊢ HasSubset.Subset (HDiv.hDiv (Set.preimage (⇑m) s) (Set.preimage (⇑m) t)) (Se …
  -/
  rintro _ ⟨_, _, _, _, rfl⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : DivisionMonoid β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    m : F
    s t : Set β
    w✝¹ : α
    left✝¹ : Membership.mem (Set.preimage (⇑m) s) w✝¹
    w✝ : α
    left✝ : Membership.mem (Set.preimage (⇑m) t) w✝
    ⊢ Membership.mem (Set.preimage (⇑m) (HDiv.hDiv s t)) ((fun x1 x2 => HDiv.hDiv  …
  -/
  exact ⟨_, ‹_›, _, ‹_›, (map_div m ..).symm⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma preimage_div (hm : Injective m) {s t : Set β} (hs : s ⊆ range m) (ht : t ⊆ range m) :
    m ⁻¹' (s / t) = m ⁻¹' s / m ⁻¹' t :=
  hm.image_injective <| by
    rw [image_div, image_preimage_eq_iff.2 hs, image_preimage_eq_iff.2 ht,
      image_preimage_eq_iff.2 (div_subset_range m hs ht)]


@[to_additive (attr := simp)]
                                                                                        /-
                                                                                          ι : Type u_5
                                                                                          α : ι → Type u_6
                                                                                          inst✝ : (i : ι) → Inv (α i)
                                                                                          s : Set ι
                                                                                          t : (i : ι) → Set (α i)
                                                                                          ⊢ Eq (Inv.inv (s.pi t)) (s.pi fun i => Inv.inv (t i))
                                                                                        -/
lemma inv_pi (s : Set ι) (t : ∀ i, Set (α i)) : (s.pi t)⁻¹ = s.pi fun i ↦ (t i)⁻¹ := by ext x; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[to_additive]
lemma MapsTo.mul [Mul β] {A : Set α} {B₁ B₂ : Set β} {f₁ f₂ : α → β}
    (h₁ : MapsTo f₁ A B₁) (h₂ : MapsTo f₂ A B₂) : MapsTo (f₁ * f₂) A (B₁ * B₂) :=
  fun _ h => mul_mem_mul (h₁ h) (h₂ h)


@[to_additive]
lemma MapsTo.inv [DivisionCommMonoid β] {A : Set α} {B : Set β} {f : α → β} (h : MapsTo f A B) :
    MapsTo (f⁻¹) A (B⁻¹) :=
  fun _ ha => inv_mem_inv.2 (h ha)



@[to_additive]
lemma MapsTo.div [DivisionCommMonoid β] {A : Set α} {B₁ B₂ : Set β} {f₁ f₂ : α → β}
    (h₁ : MapsTo f₁ A B₁) (h₂ : MapsTo f₂ A B₂) : MapsTo (f₁ / f₂) A (B₁ / B₂) :=
  fun _ ha => div_mem_div (h₁ ha) (h₂ ha)


