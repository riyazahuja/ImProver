/-- The image of a binary function `f : α → β → γ` as a function `Finset α → Finset β → Finset γ`.
Mathematically this should be thought of as the image of the corresponding function `α × β → γ`. -/
def image₂ (f : α → β → γ) (s : Finset α) (t : Finset β) : Finset γ :=
  (s ×ˢ t).image <| uncurry f


@[simp]
theorem mem_image₂ : c ∈ image₂ f s t ↔ ∃ a ∈ s, ∃ b ∈ t, f a b = c := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    c : γ
    ⊢ Iff (Membership.mem (Finset.image₂ f s t) c) (Exists fun a => And (Membershi …
  -/
  simp [image₂, and_assoc]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_image₂ (f : α → β → γ) (s : Finset α) (t : Finset β) :
    (image₂ f s t : Set γ) = Set.image2 f s t :=
  Set.ext fun _ => mem_image₂


theorem card_image₂_le (f : α → β → γ) (s : Finset α) (t : Finset β) :
    #(image₂ f s t) ≤ #s * #t :=
  card_image_le.trans_eq <| card_product _ _


theorem card_image₂_iff :
    #(image₂ f s t) = #s * #t ↔ (s ×ˢ t : Set (α × β)).InjOn fun x => f x.1 x.2 := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    ⊢ Iff (Eq (Finset.image₂ f s t).card (HMul.hMul s.card t.card)) (Set.InjOn (fu …
  -/
  rw [← card_product, ← coe_product]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    ⊢ Iff (Eq (Finset.image₂ f s t).card (SProd.sprod s t).card) (Set.InjOn (fun x …
  -/
  exact card_image_iff
  /-
    🎉 no goals
  -/


theorem card_image₂ (hf : Injective2 f) (s : Finset α) (t : Finset β) :
    #(image₂ f s t) = #s * #t :=
  (card_image_of_injective _ hf.uncurry).trans <| card_product _ _


theorem mem_image₂_of_mem (ha : a ∈ s) (hb : b ∈ t) : f a b ∈ image₂ f s t :=
  mem_image₂.2 ⟨a, ha, b, hb, rfl⟩


theorem mem_image₂_iff (hf : Injective2 f) : f a b ∈ image₂ f s t ↔ a ∈ s ∧ b ∈ t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    a : α
    b : β
    hf : Function.Injective2 f
    ⊢ Iff (Membership.mem (Finset.image₂ f s t) (f a b)) (And (Membership.mem s a) …
  -/
  rw [← mem_coe, coe_image₂, mem_image2_iff hf, mem_coe, mem_coe]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem image₂_subset (hs : s ⊆ s') (ht : t ⊆ t') : image₂ f s t ⊆ image₂ f s' t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s s' : Finset α
    t t' : Finset β
    hs : HasSubset.Subset s s'
    ht : HasSubset.Subset t t'
    ⊢ HasSubset.Subset (Finset.image₂ f s t) (Finset.image₂ f s' t')
  -/
  rw [← coe_subset, coe_image₂, coe_image₂]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s s' : Finset α
    t t' : Finset β
    hs : HasSubset.Subset s s'
    ht : HasSubset.Subset t t'
    ⊢ HasSubset.Subset (Set.image2 f ↑s ↑t) (Set.image2 f ↑s' ↑t')
  -/
  exact image2_subset hs ht
  /-
    🎉 no goals
  -/


@[gcongr]
theorem image₂_subset_left (ht : t ⊆ t') : image₂ f s t ⊆ image₂ f s t' :=
  image₂_subset Subset.rfl ht


@[gcongr]
theorem image₂_subset_right (hs : s ⊆ s') : image₂ f s t ⊆ image₂ f s' t :=
  image₂_subset hs Subset.rfl


theorem image_subset_image₂_left (hb : b ∈ t) : s.image (fun a => f a b) ⊆ image₂ f s t :=
  image_subset_iff.2 fun _ ha => mem_image₂_of_mem ha hb


theorem image_subset_image₂_right (ha : a ∈ s) : t.image (fun b => f a b) ⊆ image₂ f s t :=
  image_subset_iff.2 fun _ => mem_image₂_of_mem ha


lemma forall_mem_image₂ {p : γ → Prop} :
    (∀ z ∈ image₂ f s t, p z) ↔ ∀ x ∈ s, ∀ y ∈ t, p (f x y) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    p : γ → Prop
    ⊢ Iff (∀ (z : γ), Membership.mem (Finset.image₂ f s t) z → p z) (∀ (x : α), Me …
  -/
  simp_rw [← mem_coe, coe_image₂, forall_mem_image2]
  /-
    🎉 no goals
  -/


lemma exists_mem_image₂ {p : γ → Prop} :
    (∃ z ∈ image₂ f s t, p z) ↔ ∃ x ∈ s, ∃ y ∈ t, p (f x y) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    p : γ → Prop
    ⊢ Iff (Exists fun z => And (Membership.mem (Finset.image₂ f s t) z) (p z)) (Ex …
  -/
  simp_rw [← mem_coe, coe_image₂, exists_mem_image2]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-23")] alias forall_image₂_iff := forall_mem_image₂


@[simp]
theorem image₂_subset_iff : image₂ f s t ⊆ u ↔ ∀ x ∈ s, ∀ y ∈ t, f x y ∈ u :=
  forall_mem_image₂


theorem image₂_subset_iff_left : image₂ f s t ⊆ u ↔ ∀ a ∈ s, (t.image fun b => f a b) ⊆ u := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    u : Finset γ
    ⊢ Iff (HasSubset.Subset (Finset.image₂ f s t) u) (∀ (a : α), Membership.mem s  …
  -/
  simp_rw [image₂_subset_iff, image_subset_iff]
  /-
    🎉 no goals
  -/


theorem image₂_subset_iff_right : image₂ f s t ⊆ u ↔ ∀ b ∈ t, (s.image fun a => f a b) ⊆ u := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    u : Finset γ
    ⊢ Iff (HasSubset.Subset (Finset.image₂ f s t) u) (∀ (b : β), Membership.mem t  …
  -/
  simp_rw [image₂_subset_iff, image_subset_iff, @forall₂_swap α]
  /-
    🎉 no goals
  -/


@[simp]
theorem image₂_nonempty_iff : (image₂ f s t).Nonempty ↔ s.Nonempty ∧ t.Nonempty := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    ⊢ Iff (Finset.image₂ f s t).Nonempty (And s.Nonempty t.Nonempty)
  -/
  rw [← coe_nonempty, coe_image₂]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    ⊢ Iff (Set.image2 f ↑s ↑t).Nonempty (And s.Nonempty t.Nonempty)
  -/
  exact image2_nonempty_iff
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
theorem Nonempty.image₂ (hs : s.Nonempty) (ht : t.Nonempty) : (image₂ f s t).Nonempty :=
  image₂_nonempty_iff.2 ⟨hs, ht⟩


theorem Nonempty.of_image₂_left (h : (s.image₂ f t).Nonempty) : s.Nonempty :=
  (image₂_nonempty_iff.1 h).1


theorem Nonempty.of_image₂_right (h : (s.image₂ f t).Nonempty) : t.Nonempty :=
  (image₂_nonempty_iff.1 h).2


@[simp]
theorem image₂_empty_left : image₂ f ∅ t = ∅ :=
                      /-
                        α : Type u_1
                        β : Type u_3
                        γ : Type u_5
                        inst✝ : DecidableEq γ
                        f : α → β → γ
                        t : Finset β
                        ⊢ Eq ↑(Finset.image₂ f EmptyCollection.emptyCollection t) ↑EmptyCollection.emp …
                      -/
  coe_injective <| by simp
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem image₂_empty_right : image₂ f s ∅ = ∅ :=
                      /-
                        α : Type u_1
                        β : Type u_3
                        γ : Type u_5
                        inst✝ : DecidableEq γ
                        f : α → β → γ
                        s : Finset α
                        ⊢ Eq ↑(Finset.image₂ f s EmptyCollection.emptyCollection) ↑EmptyCollection.emp …
                      -/
  coe_injective <| by simp
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem image₂_eq_empty_iff : image₂ f s t = ∅ ↔ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    ⊢ Iff (Eq (Finset.image₂ f s t) EmptyCollection.emptyCollection) (Or (Eq s Emp …
  -/
  simp_rw [← not_nonempty_iff_eq_empty, image₂_nonempty_iff, not_and_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem image₂_singleton_left : image₂ f {a} t = t.image fun b => f a b :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    γ : Type u_5
                    inst✝ : DecidableEq γ
                    f : α → β → γ
                    t : Finset β
                    a : α
                    x : γ
                    ⊢ Iff (Membership.mem (Finset.image₂ f (Singleton.singleton a) t) x) (Membersh …
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem image₂_singleton_right : image₂ f s {b} = s.image fun a => f a b :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    γ : Type u_5
                    inst✝ : DecidableEq γ
                    f : α → β → γ
                    s : Finset α
                    b : β
                    x : γ
                    ⊢ Iff (Membership.mem (Finset.image₂ f s (Singleton.singleton b)) x) (Membersh …
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


theorem image₂_singleton_left' : image₂ f {a} t = t.image (f a) :=
  image₂_singleton_left


                                                            /-
                                                              α : Type u_1
                                                              β : Type u_3
                                                              γ : Type u_5
                                                              inst✝ : DecidableEq γ
                                                              f : α → β → γ
                                                              a : α
                                                              b : β
                                                              ⊢ Eq (Finset.image₂ f (Singleton.singleton a) (Singleton.singleton b)) (Single …
                                                            -/
theorem image₂_singleton : image₂ f {a} {b} = {f a b} := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem image₂_union_left [DecidableEq α] : image₂ f (s ∪ s') t = image₂ f s t ∪ image₂ f s' t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      ⊢ Eq ↑(Finset.image₂ f (Union.union s s') t) ↑(Union.union (Finset.image₂ f s  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      ⊢ Eq (Set.image2 f (Union.union ↑s ↑s') ↑t) (Union.union (Set.image2 f ↑s ↑t)  …
    -/
    exact image2_union_left
    /-
      🎉 no goals
    -/


theorem image₂_union_right [DecidableEq β] : image₂ f s (t ∪ t') = image₂ f s t ∪ image₂ f s t' :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t t' : Finset β
      inst✝ : DecidableEq β
      ⊢ Eq ↑(Finset.image₂ f s (Union.union t t')) ↑(Union.union (Finset.image₂ f s  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t t' : Finset β
      inst✝ : DecidableEq β
      ⊢ Eq (Set.image2 f (↑s) (Union.union ↑t ↑t')) (Union.union (Set.image2 f ↑s ↑t …
    -/
    exact image2_union_right
    /-
      🎉 no goals
    -/


@[simp]
theorem image₂_insert_left [DecidableEq α] :
    image₂ f (insert a s) t = (t.image fun b => f a b) ∪ image₂ f s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      a : α
      inst✝ : DecidableEq α
      ⊢ Eq ↑(Finset.image₂ f (Insert.insert a s) t) ↑(Union.union (Finset.image (fun …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      a : α
      inst✝ : DecidableEq α
      ⊢ Eq (Set.image2 f (Insert.insert a ↑s) ↑t) (Union.union (Set.image (fun b =>  …
    -/
    exact image2_insert_left
    /-
      🎉 no goals
    -/


@[simp]
theorem image₂_insert_right [DecidableEq β] :
    image₂ f s (insert b t) = (s.image fun a => f a b) ∪ image₂ f s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      b : β
      inst✝ : DecidableEq β
      ⊢ Eq ↑(Finset.image₂ f s (Insert.insert b t)) ↑(Union.union (Finset.image (fun …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      b : β
      inst✝ : DecidableEq β
      ⊢ Eq (Set.image2 f (↑s) (Insert.insert b ↑t)) (Union.union (Set.image (fun a = …
    -/
    exact image2_insert_right
    /-
      🎉 no goals
    -/


theorem image₂_inter_left [DecidableEq α] (hf : Injective2 f) :
    image₂ f (s ∩ s') t = image₂ f s t ∩ image₂ f s' t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      hf : Function.Injective2 f
      ⊢ Eq ↑(Finset.image₂ f (Inter.inter s s') t) ↑(Inter.inter (Finset.image₂ f s  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      hf : Function.Injective2 f
      ⊢ Eq (Set.image2 f (Inter.inter ↑s ↑s') ↑t) (Inter.inter (Set.image2 f ↑s ↑t)  …
    -/
    exact image2_inter_left hf
    /-
      🎉 no goals
    -/


theorem image₂_inter_right [DecidableEq β] (hf : Injective2 f) :
    image₂ f s (t ∩ t') = image₂ f s t ∩ image₂ f s t' :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t t' : Finset β
      inst✝ : DecidableEq β
      hf : Function.Injective2 f
      ⊢ Eq ↑(Finset.image₂ f s (Inter.inter t t')) ↑(Inter.inter (Finset.image₂ f s  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t t' : Finset β
      inst✝ : DecidableEq β
      hf : Function.Injective2 f
      ⊢ Eq (Set.image2 f (↑s) (Inter.inter ↑t ↑t')) (Inter.inter (Set.image2 f ↑s ↑t …
    -/
    exact image2_inter_right hf
    /-
      🎉 no goals
    -/


theorem image₂_inter_subset_left [DecidableEq α] :
    image₂ f (s ∩ s') t ⊆ image₂ f s t ∩ image₂ f s' t :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      ⊢ HasSubset.Subset ↑(Finset.image₂ f (Inter.inter s s') t) ↑(Inter.inter (Fins …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      ⊢ HasSubset.Subset (Set.image2 f (Inter.inter ↑s ↑s') ↑t) (Inter.inter (Set.im …
    -/
    exact image2_inter_subset_left
    /-
      🎉 no goals
    -/


theorem image₂_inter_subset_right [DecidableEq β] :
    image₂ f s (t ∩ t') ⊆ image₂ f s t ∩ image₂ f s t' :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t t' : Finset β
      inst✝ : DecidableEq β
      ⊢ HasSubset.Subset ↑(Finset.image₂ f s (Inter.inter t t')) ↑(Inter.inter (Fins …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t t' : Finset β
      inst✝ : DecidableEq β
      ⊢ HasSubset.Subset (Set.image2 f (↑s) (Inter.inter ↑t ↑t')) (Inter.inter (Set. …
    -/
    exact image2_inter_subset_right
    /-
      🎉 no goals
    -/


theorem image₂_congr (h : ∀ a ∈ s, ∀ b ∈ t, f a b = f' a b) : image₂ f s t = image₂ f' s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f f' : α → β → γ
      s : Finset α
      t : Finset β
      h : ∀ (a : α), Membership.mem s a → ∀ (b : β), Membership.mem t b → Eq (f a b) …
      ⊢ Eq ↑(Finset.image₂ f s t) ↑(Finset.image₂ f' s t)
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f f' : α → β → γ
      s : Finset α
      t : Finset β
      h : ∀ (a : α), Membership.mem s a → ∀ (b : β), Membership.mem t b → Eq (f a b) …
      ⊢ Eq (Set.image2 f ↑s ↑t) (Set.image2 f' ↑s ↑t)
    -/
    exact image2_congr h
    /-
      🎉 no goals
    -/


/-- A common special case of `image₂_congr` -/
theorem image₂_congr' (h : ∀ a b, f a b = f' a b) : image₂ f s t = image₂ f' s t :=
  image₂_congr fun a _ b _ => h a b


theorem card_image₂_singleton_left (hf : Injective (f a)) : #(image₂ f {a} t) = #t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    t : Finset β
    a : α
    hf : Function.Injective (f a)
    ⊢ Eq (Finset.image₂ f (Singleton.singleton a) t).card t.card
  -/
  rw [image₂_singleton_left, card_image_of_injective _ hf]
  /-
    🎉 no goals
  -/


theorem card_image₂_singleton_right (hf : Injective fun a => f a b) :
                                 /-
                                   α : Type u_1
                                   β : Type u_3
                                   γ : Type u_5
                                   inst✝ : DecidableEq γ
                                   f : α → β → γ
                                   s : Finset α
                                   b : β
                                   hf : Function.Injective fun a => f a b
                                   ⊢ Eq (Finset.image₂ f s (Singleton.singleton b)).card s.card
                                 -/
    #(image₂ f s {b}) = #s := by rw [image₂_singleton_right, card_image_of_injective _ hf]
                                 /-
                                   🎉 no goals
                                 -/


theorem image₂_singleton_inter [DecidableEq β] (t₁ t₂ : Finset β) (hf : Injective (f a)) :
    image₂ f {a} (t₁ ∩ t₂) = image₂ f {a} t₁ ∩ image₂ f {a} t₂ := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    a : α
    inst✝ : DecidableEq β
    t₁ t₂ : Finset β
    hf : Function.Injective (f a)
    ⊢ Eq (Finset.image₂ f (Singleton.singleton a) (Inter.inter t₁ t₂)) (Inter.inte …
  -/
  simp_rw [image₂_singleton_left, image_inter _ _ hf]
  /-
    🎉 no goals
  -/


theorem image₂_inter_singleton [DecidableEq α] (s₁ s₂ : Finset α) (hf : Injective fun a => f a b) :
    image₂ f (s₁ ∩ s₂) {b} = image₂ f s₁ {b} ∩ image₂ f s₂ {b} := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    b : β
    inst✝ : DecidableEq α
    s₁ s₂ : Finset α
    hf : Function.Injective fun a => f a b
    ⊢ Eq (Finset.image₂ f (Inter.inter s₁ s₂) (Singleton.singleton b)) (Inter.inte …
  -/
  simp_rw [image₂_singleton_right, image_inter _ _ hf]
  /-
    🎉 no goals
  -/


theorem card_le_card_image₂_left {s : Finset α} (hs : s.Nonempty) (hf : ∀ a, Injective (f a)) :
    #t ≤ #(image₂ f s t) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    t : Finset β
    s : Finset α
    hs : s.Nonempty
    hf : ∀ (a : α), Function.Injective (f a)
    ⊢ LE.le t.card (Finset.image₂ f s t).card
  -/
  obtain ⟨a, ha⟩ := hs
  /-
    case intro
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    t : Finset β
    s : Finset α
    hf : ∀ (a : α), Function.Injective (f a)
    a : α
    ha : Membership.mem s a
    ⊢ LE.le t.card (Finset.image₂ f s t).card
  -/
  rw [← card_image₂_singleton_left _ (hf a)]
  /-
    case intro
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    t : Finset β
    s : Finset α
    hf : ∀ (a : α), Function.Injective (f a)
    a : α
    ha : Membership.mem s a
    ⊢ LE.le (Finset.image₂ f (Singleton.singleton a) t).card (Finset.image₂ f s t) …
  -/
  exact card_le_card (image₂_subset_right <| singleton_subset_iff.2 ha)
  /-
    🎉 no goals
  -/


theorem card_le_card_image₂_right {t : Finset β} (ht : t.Nonempty)
    (hf : ∀ b, Injective fun a => f a b) : #s ≤ #(image₂ f s t) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    ht : t.Nonempty
    hf : ∀ (b : β), Function.Injective fun a => f a b
    ⊢ LE.le s.card (Finset.image₂ f s t).card
  -/
  obtain ⟨b, hb⟩ := ht
  /-
    case intro
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    hf : ∀ (b : β), Function.Injective fun a => f a b
    b : β
    hb : Membership.mem t b
    ⊢ LE.le s.card (Finset.image₂ f s t).card
  -/
  rw [← card_image₂_singleton_right _ (hf b)]
  /-
    case intro
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    hf : ∀ (b : β), Function.Injective fun a => f a b
    b : β
    hb : Membership.mem t b
    ⊢ LE.le (Finset.image₂ f s (Singleton.singleton b)).card (Finset.image₂ f s t) …
  -/
  exact card_le_card (image₂_subset_left <| singleton_subset_iff.2 hb)
  /-
    🎉 no goals
  -/


theorem biUnion_image_left : (s.biUnion fun a => t.image <| f a) = image₂ f s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      ⊢ Eq ↑(s.biUnion fun a => Finset.image (f a) t) ↑(Finset.image₂ f s t)
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      ⊢ Eq (Set.iUnion fun x => Set.iUnion fun x_1 => Set.image (f x) ↑t) (Set.image …
    -/
    exact Set.iUnion_image_left _
    /-
      🎉 no goals
    -/


theorem biUnion_image_right : (t.biUnion fun b => s.image fun a => f a b) = image₂ f s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      ⊢ Eq ↑(t.biUnion fun b => Finset.image (fun a => f a b) s) ↑(Finset.image₂ f s …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      ⊢ Eq (Set.iUnion fun x => Set.iUnion fun x_1 => Set.image (fun a => f a x) ↑s) …
    -/
    exact Set.iUnion_image_right _
    /-
      🎉 no goals
    -/


theorem image_image₂ (f : α → β → γ) (g : γ → δ) :
    (image₂ f s t).image g = image₂ (fun a b => g (f a b)) s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝¹ : DecidableEq γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      f : α → β → γ
      g : γ → δ
      ⊢ Eq ↑(Finset.image g (Finset.image₂ f s t)) ↑(Finset.image₂ (fun a b => g (f  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝¹ : DecidableEq γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      f : α → β → γ
      g : γ → δ
      ⊢ Eq (Set.image g (Set.image2 f ↑s ↑t)) (Set.image2 (fun a b => g (f a b)) ↑s  …
    -/
    exact image_image2 _ _
    /-
      🎉 no goals
    -/


theorem image₂_image_left (f : γ → β → δ) (g : α → γ) :
    image₂ f (s.image g) t = image₂ (fun a b => f (g a) b) s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝¹ : DecidableEq γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      f : γ → β → δ
      g : α → γ
      ⊢ Eq ↑(Finset.image₂ f (Finset.image g s) t) ↑(Finset.image₂ (fun a b => f (g  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝¹ : DecidableEq γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      f : γ → β → δ
      g : α → γ
      ⊢ Eq (Set.image2 f (Set.image g ↑s) ↑t) (Set.image2 (fun a b => f (g a) b) ↑s  …
    -/
    exact image2_image_left _ _
    /-
      🎉 no goals
    -/


theorem image₂_image_right (f : α → γ → δ) (g : β → γ) :
    image₂ f s (t.image g) = image₂ (fun a b => f a (g b)) s t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝¹ : DecidableEq γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      f : α → γ → δ
      g : β → γ
      ⊢ Eq ↑(Finset.image₂ f s (Finset.image g t)) ↑(Finset.image₂ (fun a b => f a ( …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝¹ : DecidableEq γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      f : α → γ → δ
      g : β → γ
      ⊢ Eq (Set.image2 f (↑s) (Set.image g ↑t)) (Set.image2 (fun a b => f a (g b)) ↑ …
    -/
    exact image2_image_right _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem image₂_mk_eq_product [DecidableEq α] [DecidableEq β] (s : Finset α) (t : Finset β) :
                                      /-
                                        α : Type u_1
                                        β : Type u_3
                                        inst✝¹ : DecidableEq α
                                        inst✝ : DecidableEq β
                                        s : Finset α
                                        t : Finset β
                                        ⊢ Eq (Finset.image₂ Prod.mk s t) (SProd.sprod s t)
                                      -/
    image₂ Prod.mk s t = s ×ˢ t := by ext; simp [Prod.ext_iff]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem image₂_curry (f : α × β → γ) (s : Finset α) (t : Finset β) :
    image₂ (curry f) s t = (s ×ˢ t).image f := rfl


@[simp]
theorem image_uncurry_product (f : α → β → γ) (s : Finset α) (t : Finset β) :
    (s ×ˢ t).image (uncurry f) = image₂ f s t := rfl


theorem image₂_swap (f : α → β → γ) (s : Finset α) (t : Finset β) :
    image₂ f s t = image₂ (fun a b => f b a) t s :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      ⊢ Eq ↑(Finset.image₂ f s t) ↑(Finset.image₂ (fun a b => f b a) t s)
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      ⊢ Eq (Set.image2 f ↑s ↑t) (Set.image2 (fun a b => f b a) ↑t ↑s)
    -/
    exact image2_swap _ _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem image₂_left [DecidableEq α] (h : t.Nonempty) : image₂ (fun x _ => x) s t = s :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      h : t.Nonempty
      ⊢ Eq ↑(Finset.image₂ (fun x x_1 => x) s t) ↑s
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq α
      h : t.Nonempty
      ⊢ Eq (Set.image2 (fun x x_1 => x) ↑s ↑t) ↑s
    -/
    exact image2_left h
    /-
      🎉 no goals
    -/


@[simp]
theorem image₂_right [DecidableEq β] (h : s.Nonempty) : image₂ (fun _ y => y) s t = t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq β
      h : s.Nonempty
      ⊢ Eq ↑(Finset.image₂ (fun x y => y) s t) ↑t
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq β
      h : s.Nonempty
      ⊢ Eq (Set.image2 (fun x y => y) ↑s ↑t) ↑t
    -/
    exact image2_right h
    /-
      🎉 no goals
    -/


theorem image₂_assoc {γ : Type*} {u : Finset γ}
    {f : δ → γ → ε} {g : α → β → δ} {f' : α → ε' → ε}
    {g' : β → γ → ε'} (h_assoc : ∀ a b c, f (g a b) c = f' a (g' b c)) :
    image₂ f (image₂ g s t) u = image₂ f' s (image₂ g' t u) :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      δ : Type u_7
      ε : Type u_9
      ε' : Type u_10
      inst✝² : DecidableEq ε
      inst✝¹ : DecidableEq ε'
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : δ → γ → ε
      g : α → β → δ
      f' : α → ε' → ε
      g' : β → γ → ε'
      h_assoc : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (f' a (g' b c))
      ⊢ Eq ↑(Finset.image₂ f (Finset.image₂ g s t) u) ↑(Finset.image₂ f' s (Finset.i …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      δ : Type u_7
      ε : Type u_9
      ε' : Type u_10
      inst✝² : DecidableEq ε
      inst✝¹ : DecidableEq ε'
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : δ → γ → ε
      g : α → β → δ
      f' : α → ε' → ε
      g' : β → γ → ε'
      h_assoc : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (f' a (g' b c))
      ⊢ Eq (Set.image2 f (Set.image2 g ↑s ↑t) ↑u) (Set.image2 f' (↑s) (Set.image2 g' …
    -/
    exact image2_assoc h_assoc
    /-
      🎉 no goals
    -/


theorem image₂_comm {g : β → α → γ} (h_comm : ∀ a b, f a b = g b a) : image₂ f s t = image₂ g t s :=
                                  /-
                                    α : Type u_1
                                    β : Type u_3
                                    γ : Type u_5
                                    inst✝ : DecidableEq γ
                                    f : α → β → γ
                                    s : Finset α
                                    t : Finset β
                                    g : β → α → γ
                                    h_comm : ∀ (a : α) (b : β), Eq (f a b) (g b a)
                                    ⊢ Eq (Finset.image₂ (fun a b => f b a) t s) (Finset.image₂ g t s)
                                  -/
  (image₂_swap _ _ _).trans <| by simp_rw [h_comm]
                                  /-
                                    🎉 no goals
                                  -/


theorem image₂_left_comm {γ : Type*} {u : Finset γ} {f : α → δ → ε} {g : β → γ → δ}
    {f' : α → γ → δ'} {g' : β → δ' → ε} (h_left_comm : ∀ a b c, f a (g b c) = g' b (f' a c)) :
    image₂ f s (image₂ g t u) = image₂ g' t (image₂ f' s u) :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      δ : Type u_7
      δ' : Type u_8
      ε : Type u_9
      inst✝² : DecidableEq δ'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : α → δ → ε
      g : β → γ → δ
      f' : α → γ → δ'
      g' : β → δ' → ε
      h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' b (f' a c))
      ⊢ Eq ↑(Finset.image₂ f s (Finset.image₂ g t u)) ↑(Finset.image₂ g' t (Finset.i …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      δ : Type u_7
      δ' : Type u_8
      ε : Type u_9
      inst✝² : DecidableEq δ'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : α → δ → ε
      g : β → γ → δ
      f' : α → γ → δ'
      g' : β → δ' → ε
      h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' b (f' a c))
      ⊢ Eq (Set.image2 f (↑s) (Set.image2 g ↑t ↑u)) (Set.image2 g' (↑t) (Set.image2  …
    -/
    exact image2_left_comm h_left_comm
    /-
      🎉 no goals
    -/


theorem image₂_right_comm {γ : Type*} {u : Finset γ} {f : δ → γ → ε} {g : α → β → δ}
    {f' : α → γ → δ'} {g' : δ' → β → ε} (h_right_comm : ∀ a b c, f (g a b) c = g' (f' a c) b) :
    image₂ f (image₂ g s t) u = image₂ g' (image₂ f' s u) t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      δ : Type u_7
      δ' : Type u_8
      ε : Type u_9
      inst✝² : DecidableEq δ'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : δ → γ → ε
      g : α → β → δ
      f' : α → γ → δ'
      g' : δ' → β → ε
      h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f' a c) b)
      ⊢ Eq ↑(Finset.image₂ f (Finset.image₂ g s t) u) ↑(Finset.image₂ g' (Finset.ima …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      δ : Type u_7
      δ' : Type u_8
      ε : Type u_9
      inst✝² : DecidableEq δ'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : δ → γ → ε
      g : α → β → δ
      f' : α → γ → δ'
      g' : δ' → β → ε
      h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f' a c) b)
      ⊢ Eq (Set.image2 f (Set.image2 g ↑s ↑t) ↑u) (Set.image2 g' (Set.image2 f' ↑s ↑ …
    -/
    exact image2_right_comm h_right_comm
    /-
      🎉 no goals
    -/


theorem image₂_image₂_image₂_comm {γ δ : Type*} {u : Finset γ} {v : Finset δ} [DecidableEq ζ]
    [DecidableEq ζ'] [DecidableEq ν] {f : ε → ζ → ν} {g : α → β → ε} {h : γ → δ → ζ}
    {f' : ε' → ζ' → ν} {g' : α → γ → ε'} {h' : β → δ → ζ'}
    (h_comm : ∀ a b c d, f (g a b) (h c d) = f' (g' a c) (h' b d)) :
    image₂ f (image₂ g s t) (image₂ h u v) = image₂ f' (image₂ g' s u) (image₂ h' t v) :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      ε : Type u_9
      ε' : Type u_10
      ζ : Type u_11
      ζ' : Type u_12
      ν : Type u_13
      inst✝⁴ : DecidableEq ε
      inst✝³ : DecidableEq ε'
      s : Finset α
      t : Finset β
      γ : Type u_14
      δ : Type u_15
      u : Finset γ
      v : Finset δ
      inst✝² : DecidableEq ζ
      inst✝¹ : DecidableEq ζ'
      inst✝ : DecidableEq ν
      f : ε → ζ → ν
      g : α → β → ε
      h : γ → δ → ζ
      f' : ε' → ζ' → ν
      g' : α → γ → ε'
      h' : β → δ → ζ'
      h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
      ⊢ Eq ↑(Finset.image₂ f (Finset.image₂ g s t) (Finset.image₂ h u v)) ↑(Finset.i …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      ε : Type u_9
      ε' : Type u_10
      ζ : Type u_11
      ζ' : Type u_12
      ν : Type u_13
      inst✝⁴ : DecidableEq ε
      inst✝³ : DecidableEq ε'
      s : Finset α
      t : Finset β
      γ : Type u_14
      δ : Type u_15
      u : Finset γ
      v : Finset δ
      inst✝² : DecidableEq ζ
      inst✝¹ : DecidableEq ζ'
      inst✝ : DecidableEq ν
      f : ε → ζ → ν
      g : α → β → ε
      h : γ → δ → ζ
      f' : ε' → ζ' → ν
      g' : α → γ → ε'
      h' : β → δ → ζ'
      h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
      ⊢ Eq (Set.image2 f (Set.image2 g ↑s ↑t) (Set.image2 h ↑u ↑v)) (Set.image2 f' ( …
    -/
    exact image2_image2_image2_comm h_comm
    /-
      🎉 no goals
    -/


theorem image_image₂_distrib {g : γ → δ} {f' : α' → β' → δ} {g₁ : α → α'} {g₂ : β → β'}
    (h_distrib : ∀ a b, g (f a b) = f' (g₁ a) (g₂ b)) :
    (image₂ f s t).image g = image₂ f' (s.image g₁) (t.image g₂) :=
  coe_injective <| by
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      inst✝³ : DecidableEq α'
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : α' → β' → δ
      g₁ : α → α'
      g₂ : β → β'
      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ a) (g₂ b))
      ⊢ Eq ↑(Finset.image g (Finset.image₂ f s t)) ↑(Finset.image₂ f' (Finset.image  …
    -/
    push_cast
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      inst✝³ : DecidableEq α'
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : α' → β' → δ
      g₁ : α → α'
      g₂ : β → β'
      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ a) (g₂ b))
      ⊢ Eq (Set.image g (Set.image2 f ↑s ↑t)) (Set.image2 f' (Set.image g₁ ↑s) (Set. …
    -/
    exact image_image2_distrib h_distrib
    /-
      🎉 no goals
    -/


/-- Symmetric statement to `Finset.image₂_image_left_comm`. -/
theorem image_image₂_distrib_left {g : γ → δ} {f' : α' → β → δ} {g' : α → α'}
    (h_distrib : ∀ a b, g (f a b) = f' (g' a) b) :
    (image₂ f s t).image g = image₂ f' (s.image g') t :=
  coe_injective <| by
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq α'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : α' → β → δ
      g' : α → α'
      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' a) b)
      ⊢ Eq ↑(Finset.image g (Finset.image₂ f s t)) ↑(Finset.image₂ f' (Finset.image  …
    -/
    push_cast
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq α'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : α' → β → δ
      g' : α → α'
      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' a) b)
      ⊢ Eq (Set.image g (Set.image2 f ↑s ↑t)) (Set.image2 f' (Set.image g' ↑s) ↑t)
    -/
    exact image_image2_distrib_left h_distrib
    /-
      🎉 no goals
    -/


/-- Symmetric statement to `Finset.image_image₂_right_comm`. -/
theorem image_image₂_distrib_right {g : γ → δ} {f' : α → β' → δ} {g' : β → β'}
    (h_distrib : ∀ a b, g (f a b) = f' a (g' b)) :
    (image₂ f s t).image g = image₂ f' s (t.image g') :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : α → β' → δ
      g' : β → β'
      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' a (g' b))
      ⊢ Eq ↑(Finset.image g (Finset.image₂ f s t)) ↑(Finset.image₂ f' s (Finset.imag …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : α → β' → δ
      g' : β → β'
      h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' a (g' b))
      ⊢ Eq (Set.image g (Set.image2 f ↑s ↑t)) (Set.image2 f' (↑s) (Set.image g' ↑t))
    -/
    exact image_image2_distrib_right h_distrib
    /-
      🎉 no goals
    -/


/-- Symmetric statement to `Finset.image_image₂_distrib_left`. -/
theorem image₂_image_left_comm {f : α' → β → γ} {g : α → α'} {f' : α → β → δ} {g' : δ → γ}
    (h_left_comm : ∀ a b, f (g a) b = g' (f' a b)) :
    image₂ f (s.image g) t = (image₂ f' s t).image g' :=
  (image_image₂_distrib_left fun a b => (h_left_comm a b).symm).symm


/-- Symmetric statement to `Finset.image_image₂_distrib_right`. -/
theorem image_image₂_right_comm {f : α → β' → γ} {g : β → β'} {f' : α → β → δ} {g' : δ → γ}
    (h_right_comm : ∀ a b, f a (g b) = g' (f' a b)) :
    image₂ f s (t.image g) = (image₂ f' s t).image g' :=
  (image_image₂_distrib_right fun a b => (h_right_comm a b).symm).symm


/-- The other direction does not hold because of the `s`-`s` cross terms on the RHS. -/
theorem image₂_distrib_subset_left {γ : Type*} {u : Finset γ} {f : α → δ → ε} {g : β → γ → δ}
    {f₁ : α → β → β'} {f₂ : α → γ → γ'} {g' : β' → γ' → ε}
    (h_distrib : ∀ a b c, f a (g b c) = g' (f₁ a b) (f₂ a c)) :
    image₂ f s (image₂ g t u) ⊆ image₂ g' (image₂ f₁ s t) (image₂ f₂ s u) :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ' : Type u_6
      δ : Type u_7
      ε : Type u_9
      inst✝³ : DecidableEq β'
      inst✝² : DecidableEq γ'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : α → δ → ε
      g : β → γ → δ
      f₁ : α → β → β'
      f₂ : α → γ → γ'
      g' : β' → γ' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' (f₁ a b) (f₂ a c))
      ⊢ HasSubset.Subset ↑(Finset.image₂ f s (Finset.image₂ g t u)) ↑(Finset.image₂  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ' : Type u_6
      δ : Type u_7
      ε : Type u_9
      inst✝³ : DecidableEq β'
      inst✝² : DecidableEq γ'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : α → δ → ε
      g : β → γ → δ
      f₁ : α → β → β'
      f₂ : α → γ → γ'
      g' : β' → γ' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' (f₁ a b) (f₂ a c))
      ⊢ HasSubset.Subset (Set.image2 f (↑s) (Set.image2 g ↑t ↑u)) (Set.image2 g' (Se …
    -/
    exact Set.image2_distrib_subset_left h_distrib
    /-
      🎉 no goals
    -/


/-- The other direction does not hold because of the `u`-`u` cross terms on the RHS. -/
theorem image₂_distrib_subset_right {γ : Type*} {u : Finset γ} {f : δ → γ → ε} {g : α → β → δ}
    {f₁ : α → γ → α'} {f₂ : β → γ → β'} {g' : α' → β' → ε}
    (h_distrib : ∀ a b c, f (g a b) c = g' (f₁ a c) (f₂ b c)) :
    image₂ f (image₂ g s t) u ⊆ image₂ g' (image₂ f₁ s u) (image₂ f₂ t u) :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      δ : Type u_7
      ε : Type u_9
      inst✝³ : DecidableEq α'
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : δ → γ → ε
      g : α → β → δ
      f₁ : α → γ → α'
      f₂ : β → γ → β'
      g' : α' → β' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f₁ a c) (f₂ b c))
      ⊢ HasSubset.Subset ↑(Finset.image₂ f (Finset.image₂ g s t) u) ↑(Finset.image₂  …
    -/
    push_cast
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      β' : Type u_4
      δ : Type u_7
      ε : Type u_9
      inst✝³ : DecidableEq α'
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq ε
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      γ : Type u_14
      u : Finset γ
      f : δ → γ → ε
      g : α → β → δ
      f₁ : α → γ → α'
      f₂ : β → γ → β'
      g' : α' → β' → ε
      h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f₁ a c) (f₂ b c))
      ⊢ HasSubset.Subset (Set.image2 f (Set.image2 g ↑s ↑t) ↑u) (Set.image2 g' (Set. …
    -/
    exact Set.image2_distrib_subset_right h_distrib
    /-
      🎉 no goals
    -/


theorem image_image₂_antidistrib {g : γ → δ} {f' : β' → α' → δ} {g₁ : β → β'} {g₂ : α → α'}
    (h_antidistrib : ∀ a b, g (f a b) = f' (g₁ b) (g₂ a)) :
    (image₂ f s t).image g = image₂ f' (t.image g₁) (s.image g₂) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    inst✝³ : DecidableEq α'
    inst✝² : DecidableEq β'
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq δ
    g : γ → δ
    f' : β' → α' → δ
    g₁ : β → β'
    g₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ b) (g₂ a))
    ⊢ Eq (Finset.image g (Finset.image₂ f s t)) (Finset.image₂ f' (Finset.image g₁ …
  -/
  rw [image₂_swap f]
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    inst✝³ : DecidableEq α'
    inst✝² : DecidableEq β'
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq δ
    g : γ → δ
    f' : β' → α' → δ
    g₁ : β → β'
    g₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ b) (g₂ a))
    ⊢ Eq (Finset.image g (Finset.image₂ (fun a b => f b a) t s)) (Finset.image₂ f' …
  -/
  exact image_image₂_distrib fun _ _ => h_antidistrib _ _
  /-
    🎉 no goals
  -/


/-- Symmetric statement to `Finset.image₂_image_left_anticomm`. -/
theorem image_image₂_antidistrib_left {g : γ → δ} {f' : β' → α → δ} {g' : β → β'}
    (h_antidistrib : ∀ a b, g (f a b) = f' (g' b) a) :
    (image₂ f s t).image g = image₂ f' (t.image g') s :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : β' → α → δ
      g' : β → β'
      h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' b) a)
      ⊢ Eq ↑(Finset.image g (Finset.image₂ f s t)) ↑(Finset.image₂ f' (Finset.image  …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      β' : Type u_4
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq β'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : β' → α → δ
      g' : β → β'
      h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' b) a)
      ⊢ Eq (Set.image g (Set.image2 f ↑s ↑t)) (Set.image2 f' (Set.image g' ↑t) ↑s)
    -/
    exact image_image2_antidistrib_left h_antidistrib
    /-
      🎉 no goals
    -/


/-- Symmetric statement to `Finset.image_image₂_right_anticomm`. -/
theorem image_image₂_antidistrib_right {g : γ → δ} {f' : β → α' → δ} {g' : α → α'}
    (h_antidistrib : ∀ a b, g (f a b) = f' b (g' a)) :
    (image₂ f s t).image g = image₂ f' t (s.image g') :=
  coe_injective <| by
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq α'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : β → α' → δ
      g' : α → α'
      h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' b (g' a))
      ⊢ Eq ↑(Finset.image g (Finset.image₂ f s t)) ↑(Finset.image₂ f' t (Finset.imag …
    -/
    push_cast
    /-
      α : Type u_1
      α' : Type u_2
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      inst✝² : DecidableEq α'
      inst✝¹ : DecidableEq γ
      f : α → β → γ
      s : Finset α
      t : Finset β
      inst✝ : DecidableEq δ
      g : γ → δ
      f' : β → α' → δ
      g' : α → α'
      h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' b (g' a))
      ⊢ Eq (Set.image g (Set.image2 f ↑s ↑t)) (Set.image2 f' (↑t) (Set.image g' ↑s))
    -/
    exact image_image2_antidistrib_right h_antidistrib
    /-
      🎉 no goals
    -/


/-- Symmetric statement to `Finset.image_image₂_antidistrib_left`. -/
theorem image₂_image_left_anticomm {f : α' → β → γ} {g : α → α'} {f' : β → α → δ} {g' : δ → γ}
    (h_left_anticomm : ∀ a b, f (g a) b = g' (f' b a)) :
    image₂ f (s.image g) t = (image₂ f' t s).image g' :=
  (image_image₂_antidistrib_left fun a b => (h_left_anticomm b a).symm).symm


/-- Symmetric statement to `Finset.image_image₂_antidistrib_right`. -/
theorem image_image₂_right_anticomm {f : α → β' → γ} {g : β → β'} {f' : β → α → δ} {g' : δ → γ}
    (h_right_anticomm : ∀ a b, f a (g b) = g' (f' b a)) :
    image₂ f s (t.image g) = (image₂ f' t s).image g' :=
  (image_image₂_antidistrib_right fun a b => (h_right_anticomm b a).symm).symm


/-- If `a` is a left identity for `f : α → β → β`, then `{a}` is a left identity for
`Finset.image₂ f`. -/
theorem image₂_left_identity {f : α → γ → γ} {a : α} (h : ∀ b, f a b = b) (t : Finset γ) :
    image₂ f {a} t = t :=
                      /-
                        α : Type u_1
                        γ : Type u_5
                        inst✝ : DecidableEq γ
                        f : α → γ → γ
                        a : α
                        h : ∀ (b : γ), Eq (f a b) b
                        t : Finset γ
                        ⊢ Eq ↑(Finset.image₂ f (Singleton.singleton a) t) ↑t
                      -/
  coe_injective <| by rw [coe_image₂, coe_singleton, Set.image2_left_identity h]
                      /-
                        🎉 no goals
                      -/


/-- If `b` is a right identity for `f : α → β → α`, then `{b}` is a right identity for
`Finset.image₂ f`. -/
theorem image₂_right_identity {f : γ → β → γ} {b : β} (h : ∀ a, f a b = a) (s : Finset γ) :
                             /-
                               β : Type u_3
                               γ : Type u_5
                               inst✝ : DecidableEq γ
                               f : γ → β → γ
                               b : β
                               h : ∀ (a : γ), Eq (f a b) a
                               s : Finset γ
                               ⊢ Eq (Finset.image₂ f s (Singleton.singleton b)) s
                             -/
    image₂ f s {b} = s := by rw [image₂_singleton_right, funext h, image_id']
                             /-
                               🎉 no goals
                             -/


/-- If each partial application of `f` is injective, and images of `s` under those partial
applications are disjoint (but not necessarily distinct!), then the size of `t` divides the size of
`Finset.image₂ f s t`. -/
theorem card_dvd_card_image₂_right (hf : ∀ a ∈ s, Injective (f a))
    (hs : ((fun a => t.image <| f a) '' s).PairwiseDisjoint id) : #t ∣ #(image₂ f s t) := by
  classical
  induction' s using Finset.induction with a s _ ih
  · simp
  specialize ih (forall_of_forall_insert hf)
    (hs.subset <| Set.image_subset _ <| coe_subset.2 <| subset_insert _ _)
  rw [image₂_insert_left]
  by_cases h : Disjoint (image (f a) t) (image₂ f s t)
  · rw [card_union_of_disjoint h]
    exact Nat.dvd_add (card_image_of_injective _ <| hf _ <| mem_insert_self _ _).symm.dvd ih
  simp_rw [← biUnion_image_left, disjoint_biUnion_right, not_forall] at h
  obtain ⟨b, hb, h⟩ := h
  rwa [union_eq_right.2]
  exact (hs.eq (Set.mem_image_of_mem _ <| mem_insert_self _ _)
      (Set.mem_image_of_mem _ <| mem_insert_of_mem hb) h).trans_subset
    (image_subset_image₂_right hb)


/-- If each partial application of `f` is injective, and images of `t` under those partial
applications are disjoint (but not necessarily distinct!), then the size of `s` divides the size of
`Finset.image₂ f s t`. -/
theorem card_dvd_card_image₂_left (hf : ∀ b ∈ t, Injective fun a => f a b)
    (ht : ((fun b => s.image fun a => f a b) '' t).PairwiseDisjoint id) :
                               /-
                                 α : Type u_1
                                 β : Type u_3
                                 γ : Type u_5
                                 inst✝ : DecidableEq γ
                                 f : α → β → γ
                                 s : Finset α
                                 t : Finset β
                                 hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
                                 ht : (Set.image (fun b => Finset.image (fun a => f a b) s) ↑t).PairwiseDisjoin …
                                 ⊢ Dvd.dvd s.card (Finset.image₂ f s t).card
                               -/
    #s ∣ #(image₂ f s t) := by rw [← image₂_swap]; exact card_dvd_card_image₂_right hf ht
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If a `Finset` is a subset of the image of two `Set`s under a binary operation,
then it is a subset of the `Finset.image₂` of two `Finset` subsets of these `Set`s. -/
theorem subset_set_image₂ {s : Set α} {t : Set β} (hu : ↑u ⊆ image2 f s t) :
    ∃ (s' : Finset α) (t' : Finset β), ↑s' ⊆ s ∧ ↑t' ⊆ t ∧ u ⊆ image₂ f s' t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    u : Finset γ
    s : Set α
    t : Set β
    hu : HasSubset.Subset (↑u) (Set.image2 f s t)
    ⊢ Exists fun s' => Exists fun t' => And (HasSubset.Subset (↑s') s) (And (HasSu …
  -/
  rw [← Set.image_prod, subset_set_image_iff] at hu
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    inst✝ : DecidableEq γ
    f : α → β → γ
    u : Finset γ
    s : Set α
    t : Set β
    hu : Exists fun s' => And (HasSubset.Subset (↑s') (SProd.sprod s t)) (Eq (Fins …
    ⊢ Exists fun s' => Exists fun t' => And (HasSubset.Subset (↑s') s) (And (HasSu …
  -/
  rcases hu with ⟨u, hu, rfl⟩
  classical
  use u.image Prod.fst, u.image Prod.snd
  simp only [coe_image, Set.image_subset_iff, image₂_image_left, image₂_image_right,
    image_subset_iff]
  exact ⟨fun _ h ↦ (hu h).1, fun _ h ↦ (hu h).2, fun x hx ↦ mem_image₂_of_mem hx hx⟩


@[deprecated (since := "2024-09-22")] alias subset_image₂ := subset_set_image₂


theorem image₂_inter_union_subset_union :
    image₂ f (s ∩ s') (t ∪ t') ⊆ image₂ f s t ∪ image₂ f s' t' :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝² : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t t' : Finset β
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      ⊢ HasSubset.Subset ↑(Finset.image₂ f (Inter.inter s s') (Union.union t t')) ↑( …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝² : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t t' : Finset β
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      ⊢ HasSubset.Subset (Set.image2 f (Inter.inter ↑s ↑s') (Union.union ↑t ↑t')) (U …
    -/
    exact Set.image2_inter_union_subset_union
    /-
      🎉 no goals
    -/


theorem image₂_union_inter_subset_union :
    image₂ f (s ∪ s') (t ∩ t') ⊆ image₂ f s t ∪ image₂ f s' t' :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝² : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t t' : Finset β
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      ⊢ HasSubset.Subset ↑(Finset.image₂ f (Union.union s s') (Inter.inter t t')) ↑( …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      inst✝² : DecidableEq γ
      f : α → β → γ
      s s' : Finset α
      t t' : Finset β
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      ⊢ HasSubset.Subset (Set.image2 f (Union.union ↑s ↑s') (Inter.inter ↑t ↑t')) (U …
    -/
    exact Set.image2_union_inter_subset_union
    /-
      🎉 no goals
    -/


theorem image₂_inter_union_subset {f : α → α → β} {s t : Finset α} (hf : ∀ a b, f a b = f b a) :
    image₂ f (s ∩ t) (s ∪ t) ⊆ image₂ f s t :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → α → β
      s t : Finset α
      hf : ∀ (a b : α), Eq (f a b) (f b a)
      ⊢ HasSubset.Subset ↑(Finset.image₂ f (Inter.inter s t) (Union.union s t)) ↑(Fi …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → α → β
      s t : Finset α
      hf : ∀ (a b : α), Eq (f a b) (f b a)
      ⊢ HasSubset.Subset (Set.image2 f (Inter.inter ↑s ↑t) (Union.union ↑s ↑t)) (Set …
    -/
    exact image2_inter_union_subset hf
    /-
      🎉 no goals
    -/


theorem image₂_union_inter_subset {f : α → α → β} {s t : Finset α} (hf : ∀ a b, f a b = f b a) :
    image₂ f (s ∪ t) (s ∩ t) ⊆ image₂ f s t :=
  coe_subset.1 <| by
    /-
      α : Type u_1
      β : Type u_3
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → α → β
      s t : Finset α
      hf : ∀ (a b : α), Eq (f a b) (f b a)
      ⊢ HasSubset.Subset ↑(Finset.image₂ f (Union.union s t) (Inter.inter s t)) ↑(Fi …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_3
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → α → β
      s t : Finset α
      hf : ∀ (a b : α), Eq (f a b) (f b a)
      ⊢ HasSubset.Subset (Set.image2 f (Union.union ↑s ↑t) (Inter.inter ↑s ↑t)) (Set …
    -/
    exact image2_union_inter_subset hf
    /-
      🎉 no goals
    -/


@[simp (default + 1)] -- otherwise `simp` doesn't use `forall_mem_image₂`
lemma sup'_image₂_le {g : γ → δ} {a : δ} (h : (image₂ f s t).Nonempty) :
    sup' (image₂ f s t) h g ≤ a ↔ ∀ x ∈ s, ∀ y ∈ t, g (f x y) ≤ a := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝ : SemilatticeSup δ
    g : γ → δ
    a : δ
    h : (Finset.image₂ f s t).Nonempty
    ⊢ Iff (LE.le ((Finset.image₂ f s t).sup' h g) a) (∀ (x : α), Membership.mem s  …
  -/
  rw [sup'_le_iff, forall_mem_image₂]
  /-
    🎉 no goals
  -/


lemma sup'_image₂_left (g : γ → δ) (h : (image₂ f s t).Nonempty) :
    sup' (image₂ f s t) h g =
      sup' s h.of_image₂_left fun x ↦ sup' t h.of_image₂_right (g <| f x ·) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝ : SemilatticeSup δ
    g : γ → δ
    h : (Finset.image₂ f s t).Nonempty
    ⊢ Eq ((Finset.image₂ f s t).sup' h g) (s.sup' ⋯ fun x => t.sup' ⋯ fun x_1 => g …
  -/
  simp only [image₂, sup'_image, sup'_product_left]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma sup'_image₂_right (g : γ → δ) (h : (image₂ f s t).Nonempty) :
    sup' (image₂ f s t) h g =
      sup' t h.of_image₂_right fun y ↦ sup' s h.of_image₂_left (g <| f · y) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝ : SemilatticeSup δ
    g : γ → δ
    h : (Finset.image₂ f s t).Nonempty
    ⊢ Eq ((Finset.image₂ f s t).sup' h g) (t.sup' ⋯ fun y => s.sup' ⋯ fun x => g ( …
  -/
  simp only [image₂, sup'_image, sup'_product_right]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp (default + 1)] -- otherwise `simp` doesn't use `forall_mem_image₂`
lemma sup_image₂_le {g : γ → δ} {a : δ} :
    sup (image₂ f s t) g ≤ a ↔ ∀ x ∈ s, ∀ y ∈ t, g (f x y) ≤ a := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝² : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝¹ : SemilatticeSup δ
    inst✝ : OrderBot δ
    g : γ → δ
    a : δ
    ⊢ Iff (LE.le ((Finset.image₂ f s t).sup g) a) (∀ (x : α), Membership.mem s x → …
  -/
  rw [Finset.sup_le_iff, forall_mem_image₂]
  /-
    🎉 no goals
  -/


lemma sup_image₂_left (g : γ → δ) : sup (image₂ f s t) g = sup s fun x ↦ sup t (g <| f x ·) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝² : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝¹ : SemilatticeSup δ
    inst✝ : OrderBot δ
    g : γ → δ
    ⊢ Eq ((Finset.image₂ f s t).sup g) (s.sup fun x => t.sup fun x_1 => g (f x x_1))
  -/
  simp only [image₂, sup_image, sup_product_left]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma sup_image₂_right (g : γ → δ) : sup (image₂ f s t) g = sup t fun y ↦ sup s (g <| f · y) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝² : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝¹ : SemilatticeSup δ
    inst✝ : OrderBot δ
    g : γ → δ
    ⊢ Eq ((Finset.image₂ f s t).sup g) (t.sup fun y => s.sup fun x => g (f x y))
  -/
  simp only [image₂, sup_image, sup_product_right]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp (default + 1)] -- otherwise `simp` doesn't use `forall_mem_image₂`
lemma le_inf'_image₂ {g : γ → δ} {a : δ} (h : (image₂ f s t).Nonempty) :
    a ≤ inf' (image₂ f s t) h g ↔ ∀ x ∈ s, ∀ y ∈ t, a ≤ g (f x y) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    inst✝¹ : DecidableEq γ
    f : α → β → γ
    s : Finset α
    t : Finset β
    inst✝ : SemilatticeInf δ
    g : γ → δ
    a : δ
    h : (Finset.image₂ f s t).Nonempty
    ⊢ Iff (LE.le a ((Finset.image₂ f s t).inf' h g)) (∀ (x : α), Membership.mem s  …
  -/
  rw [le_inf'_iff, forall_mem_image₂]
  /-
    🎉 no goals
  -/


lemma inf'_image₂_left (g : γ → δ) (h : (image₂ f s t).Nonempty) :
    inf' (image₂ f s t) h g =
      inf' s h.of_image₂_left fun x ↦ inf' t h.of_image₂_right (g <| f x ·) :=
  sup'_image₂_left (δ := δᵒᵈ) g h


lemma inf'_image₂_right (g : γ → δ) (h : (image₂ f s t).Nonempty) :
    inf' (image₂ f s t) h g =
      inf' t h.of_image₂_right fun y ↦ inf' s h.of_image₂_left (g <| f · y) :=
  sup'_image₂_right (δ := δᵒᵈ) g h


@[simp (default + 1)] -- otherwise `simp` doesn't use `forall_mem_image₂`
lemma le_inf_image₂ {g : γ → δ} {a : δ} :
    a ≤ inf (image₂ f s t) g ↔ ∀ x ∈ s, ∀ y ∈ t, a ≤ g (f x y) :=
  sup_image₂_le (δ := δᵒᵈ)


lemma inf_image₂_left (g : γ → δ) : inf (image₂ f s t) g = inf s fun x ↦ inf t (g ∘ f x) :=
  sup_image₂_left (δ := δᵒᵈ) ..


lemma inf_image₂_right (g : γ → δ) : inf (image₂ f s t) g = inf t fun y ↦ inf s (g <| f · y) :=
  sup_image₂_right (δ := δᵒᵈ) ..


lemma piFinset_image₂ (f : ∀ i, α i → β i → γ i) (s : ∀ i, Finset (α i)) (t : ∀ i, Finset (β i)) :
    piFinset (fun i ↦ image₂ (f i) (s i) (t i)) =
      image₂ (fun a b i ↦ f _ (a i) (b i)) (piFinset s) (piFinset t) := by
  /-
    ι : Type u_14
    α : ι → Type u_15
    β : ι → Type u_16
    γ : ι → Type u_17
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (γ i)
    f : (i : ι) → α i → β i → γ i
    s : (i : ι) → Finset (α i)
    t : (i : ι) → Finset (β i)
    ⊢ Eq (Fintype.piFinset fun i => Finset.image₂ (f i) (s i) (t i)) (Finset.image …
  -/
  ext; simp only [mem_piFinset, mem_image₂, Classical.skolem, forall_and, funext_iff]
       /-
         🎉 no goals
       -/


@[simp]
theorem toFinset_image2 (f : α → β → γ) (s : Set α) (t : Set β) [Fintype s] [Fintype t]
    [Fintype (image2 f s t)] : (image2 f s t).toFinset = Finset.image₂ f s.toFinset t.toFinset :=
                             /-
                               α : Type u_1
                               β : Type u_3
                               γ : Type u_5
                               inst✝³ : DecidableEq γ
                               f : α → β → γ
                               s : Set α
                               t : Set β
                               inst✝² : Fintype ↑s
                               inst✝¹ : Fintype ↑t
                               inst✝ : Fintype ↑(Set.image2 f s t)
                               ⊢ Eq ↑(Set.image2 f s t).toFinset ↑(Finset.image₂ f s.toFinset t.toFinset)
                             -/
  Finset.coe_injective <| by simp
                             /-
                               🎉 no goals
                             -/


theorem Finite.toFinset_image2 (f : α → β → γ) (hs : s.Finite) (ht : t.Finite)
    (hf := hs.image2 f ht) : hf.toFinset = Finset.image₂ f hs.toFinset ht.toFinset :=
                             /-
                               α : Type u_1
                               β : Type u_3
                               γ : Type u_5
                               inst✝ : DecidableEq γ
                               s : Set α
                               t : Set β
                               f : α → β → γ
                               hs : s.Finite
                               ht : t.Finite
                               hf : optParam (Set.image2 f s t).Finite ⋯
                               ⊢ Eq ↑(Set.Finite.toFinset hf) ↑(Finset.image₂ f hs.toFinset ht.toFinset)
                             -/
  Finset.coe_injective <| by simp
                             /-
                               🎉 no goals
                             -/


