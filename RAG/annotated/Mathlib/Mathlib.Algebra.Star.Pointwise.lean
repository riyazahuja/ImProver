local postfix:max "⋆" => star


/-- The set `(star s : Set α)` is defined as `{x | star x ∈ s}` in the locale `Pointwise`.
In the usual case where `star` is involutive, it is equal to `{star s | x ∈ s}`, see
`Set.image_star`. -/
protected def star [Star α] : Star (Set α) := ⟨preimage Star.star⟩


@[simp]
theorem star_empty [Star α] : (∅ : Set α)⋆ = ∅ := rfl


@[simp]
theorem star_univ [Star α] : (univ : Set α)⋆ = univ := rfl


@[simp]
theorem nonempty_star [InvolutiveStar α] {s : Set α} : s⋆.Nonempty ↔ s.Nonempty :=
  star_involutive.surjective.nonempty_preimage


theorem Nonempty.star [InvolutiveStar α] {s : Set α} (h : s.Nonempty) : s⋆.Nonempty :=
  nonempty_star.2 h


@[simp]
theorem mem_star [Star α] : a ∈ s⋆ ↔ a⋆ ∈ s := Iff.rfl


                                                                 /-
                                                                   α : Type u_1
                                                                   s : Set α
                                                                   a : α
                                                                   inst✝ : InvolutiveStar α
                                                                   ⊢ Iff (Membership.mem (Star.star s) (Star.star a)) (Membership.mem s a)
                                                                 -/
theorem star_mem_star [InvolutiveStar α] : a⋆ ∈ s⋆ ↔ a ∈ s := by simp only [mem_star, star_star]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem star_preimage [Star α] : Star.star ⁻¹' s = s⋆ := rfl


@[simp]
theorem image_star [InvolutiveStar α] : Star.star '' s = s⋆ := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : InvolutiveStar α
    ⊢ Eq (Set.image Star.star s) (Star.star s)
  -/
  simp only [← star_preimage]
  /-
    α : Type u_1
    s : Set α
    inst✝ : InvolutiveStar α
    ⊢ Eq (Set.image Star.star s) (Set.preimage Star.star s)
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  rw [image_eq_preimage_of_inverse] <;> intro <;> simp only [star_star]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem inter_star [Star α] : (s ∩ t)⋆ = s⋆ ∩ t⋆ := preimage_inter


@[simp]
theorem union_star [Star α] : (s ∪ t)⋆ = s⋆ ∪ t⋆ := preimage_union


@[simp]
theorem iInter_star {ι : Sort*} [Star α] (s : ι → Set α) : (⋂ i, s i)⋆ = ⋂ i, (s i)⋆ :=
  preimage_iInter


@[simp]
theorem iUnion_star {ι : Sort*} [Star α] (s : ι → Set α) : (⋃ i, s i)⋆ = ⋃ i, (s i)⋆ :=
  preimage_iUnion


@[simp]
theorem compl_star [Star α] : sᶜ⋆ = s⋆ᶜ := preimage_compl


@[simp]
instance [InvolutiveStar α] : InvolutiveStar (Set α) where
  star := Star.star
                          /-
                            α : Type u_1
                            s✝ t : Set α
                            a : α
                            inst✝ : InvolutiveStar α
                            s : Set α
                            ⊢ Eq (Star.star (Star.star s)) s
                          -/
  star_involutive s := by simp only [← star_preimage, preimage_preimage, star_star, preimage_id']
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem star_subset_star [InvolutiveStar α] {s t : Set α} : s⋆ ⊆ t⋆ ↔ s ⊆ t :=
  Equiv.star.surjective.preimage_subset_preimage_iff


theorem star_subset [InvolutiveStar α] {s t : Set α} : s⋆ ⊆ t ↔ s ⊆ t⋆ := by
  /-
    α : Type u_1
    inst✝ : InvolutiveStar α
    s t : Set α
    ⊢ Iff (HasSubset.Subset (Star.star s) t) (HasSubset.Subset s (Star.star t))
  -/
  rw [← star_subset_star, star_star]
  /-
    🎉 no goals
  -/


theorem Finite.star [InvolutiveStar α] {s : Set α} (hs : s.Finite) : s⋆.Finite :=
  hs.preimage star_injective.injOn


theorem star_singleton {β : Type*} [InvolutiveStar β] (x : β) : ({x} : Set β)⋆ = {x⋆} := by
  /-
    β : Type u_2
    inst✝ : InvolutiveStar β
    x : β
    ⊢ Eq (Star.star (Singleton.singleton x)) (Singleton.singleton (Star.star x))
  -/
  ext1 y
  /-
    case h
    β : Type u_2
    inst✝ : InvolutiveStar β
    x y : β
    ⊢ Iff (Membership.mem (Star.star (Singleton.singleton x)) y) (Membership.mem ( …
  -/
  rw [mem_star, mem_singleton_iff, mem_singleton_iff, star_eq_iff_star_eq, eq_comm]
  /-
    🎉 no goals
  -/


protected theorem star_mul [Mul α] [StarMul α] (s t : Set α) : (s * t)⋆ = t⋆ * s⋆ := by
 simp_rw [← image_star, ← image2_mul, image_image2, image2_image_left, image2_image_right,
   star_mul, image2_swap _ s t]


protected theorem star_add [AddMonoid α] [StarAddMonoid α] (s t : Set α) : (s + t)⋆ = s⋆ + t⋆ := by
 simp_rw [← image_star, ← image2_add, image_image2, image2_image_left, image2_image_right,
   star_add]


@[simp]
instance [Star α] [TrivialStar α] : TrivialStar (Set α) where
  star_trivial s := by
    /-
      α : Type u_1
      s✝ t : Set α
      a : α
      inst✝¹ : Star α
      inst✝ : TrivialStar α
      s : Set α
      ⊢ Eq (Star.star s) s
    -/
    rw [← star_preimage]
    /-
      α : Type u_1
      s✝ t : Set α
      a : α
      inst✝¹ : Star α
      inst✝ : TrivialStar α
      s : Set α
      ⊢ Eq (Set.preimage Star.star s) s
    -/
    ext1
    /-
      case h
      α : Type u_1
      s✝ t : Set α
      a : α
      inst✝¹ : Star α
      inst✝ : TrivialStar α
      s : Set α
      x✝ : α
      ⊢ Iff (Membership.mem (Set.preimage Star.star s) x✝) (Membership.mem s x✝)
    -/
    simp [star_trivial]
    /-
      🎉 no goals
    -/


protected theorem star_inv [Group α] [StarMul α] (s : Set α) : s⁻¹⋆ = s⋆⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : StarMul α
    s : Set α
    ⊢ Eq (Star.star (Inv.inv s)) (Inv.inv (Star.star s))
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : StarMul α
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (Star.star (Inv.inv s)) x✝) (Membership.mem (Inv.inv (St …
  -/
  simp only [mem_star, mem_inv, star_inv]
  /-
    🎉 no goals
  -/


protected theorem star_inv' [DivisionSemiring α] [StarRing α] (s : Set α) : s⁻¹⋆ = s⋆⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : DivisionSemiring α
    inst✝ : StarRing α
    s : Set α
    ⊢ Eq (Star.star (Inv.inv s)) (Inv.inv (Star.star s))
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : DivisionSemiring α
    inst✝ : StarRing α
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (Star.star (Inv.inv s)) x✝) (Membership.mem (Inv.inv (St …
  -/
  simp only [mem_star, mem_inv, star_inv₀]
  /-
    🎉 no goals
  -/


@[simp]
lemma StarMemClass.star_coe_eq {S α : Type*} [InvolutiveStar α] [SetLike S α]
    [StarMemClass S α] (s : S) : star (s : Set α) = s := by
  /-
    S : Type u_1
    α : Type u_2
    inst✝² : InvolutiveStar α
    inst✝¹ : SetLike S α
    inst✝ : StarMemClass S α
    s : S
    ⊢ Eq (Star.star ↑s) ↑s
  -/
  ext x
  /-
    case h
    S : Type u_1
    α : Type u_2
    inst✝² : InvolutiveStar α
    inst✝¹ : SetLike S α
    inst✝ : StarMemClass S α
    s : S
    x : α
    ⊢ Iff (Membership.mem (Star.star ↑s) x) (Membership.mem (↑s) x)
  -/
  simp only [Set.mem_star, SetLike.mem_coe]
  /-
    case h
    S : Type u_1
    α : Type u_2
    inst✝² : InvolutiveStar α
    inst✝¹ : SetLike S α
    inst✝ : StarMemClass S α
    s : S
    x : α
    ⊢ Iff (Membership.mem s (Star.star x)) (Membership.mem s x)
  -/
  exact ⟨by simpa only [star_star] using star_mem (s := s) (r := star x), star_mem⟩
  /-
    🎉 no goals
  -/

