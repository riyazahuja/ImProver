theorem mem_image2_iff (hf : Injective2 f) : f a b ∈ image2 f s t ↔ a ∈ s ∧ b ∈ t :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      f : α → β → γ
      s : Set α
      t : Set β
      a : α
      b : β
      hf : Function.Injective2 f
      ⊢ Membership.mem (Set.image2 f s t) (f a b) → And (Membership.mem s a) (Member …
    -/
    rintro ⟨a', ha', b', hb', h⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      f : α → β → γ
      s : Set α
      t : Set β
      a : α
      b : β
      hf : Function.Injective2 f
      a' : α
      ha' : Membership.mem s a'
      b' : β
      hb' : Membership.mem t b'
      h : Eq (f a' b') (f a b)
      ⊢ And (Membership.mem s a) (Membership.mem t b)
    -/
    rcases hf h with ⟨rfl, rfl⟩
    /-
      case intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      f : α → β → γ
      s : Set α
      t : Set β
      hf : Function.Injective2 f
      a' : α
      ha' : Membership.mem s a'
      b' : β
      hb' : Membership.mem t b'
      h : Eq (f a' b') (f a' b')
      ⊢ And (Membership.mem s a') (Membership.mem t b')
    -/
    exact ⟨ha', hb'⟩, fun ⟨ha, hb⟩ => mem_image2_of_mem ha hb⟩
    /-
      🎉 no goals
    -/


/-- image2 is monotone with respect to `⊆`. -/
@[gcongr]
theorem image2_subset (hs : s ⊆ s') (ht : t ⊆ t') : image2 f s t ⊆ image2 f s' t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s s' : Set α
    t t' : Set β
    hs : HasSubset.Subset s s'
    ht : HasSubset.Subset t t'
    ⊢ HasSubset.Subset (Set.image2 f s t) (Set.image2 f s' t')
  -/
  rintro _ ⟨a, ha, b, hb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s s' : Set α
    t t' : Set β
    hs : HasSubset.Subset s s'
    ht : HasSubset.Subset t t'
    a : α
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    ⊢ Membership.mem (Set.image2 f s' t') (f a b)
  -/
  exact mem_image2_of_mem (hs ha) (ht hb)
  /-
    🎉 no goals
  -/


@[gcongr]
theorem image2_subset_left (ht : t ⊆ t') : image2 f s t ⊆ image2 f s t' :=
  image2_subset Subset.rfl ht


@[gcongr]
theorem image2_subset_right (hs : s ⊆ s') : image2 f s t ⊆ image2 f s' t :=
  image2_subset hs Subset.rfl


theorem image_subset_image2_left (hb : b ∈ t) : (fun a => f a b) '' s ⊆ image2 f s t :=
  forall_mem_image.2 fun _ ha => mem_image2_of_mem ha hb


theorem image_subset_image2_right (ha : a ∈ s) : f a '' t ⊆ image2 f s t :=
  forall_mem_image.2 fun _ => mem_image2_of_mem ha


lemma forall_mem_image2 {p : γ → Prop} :
                                                                  /-
                                                                    α : Type u_1
                                                                    β : Type u_3
                                                                    γ : Type u_5
                                                                    f : α → β → γ
                                                                    s : Set α
                                                                    t : Set β
                                                                    p : γ → Prop
                                                                    ⊢ Iff (∀ (z : γ), Membership.mem (Set.image2 f s t) z → p z) (∀ (x : α), Membe …
                                                                  -/
    (∀ z ∈ image2 f s t, p z) ↔ ∀ x ∈ s, ∀ y ∈ t, p (f x y) := by aesop
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma exists_mem_image2 {p : γ → Prop} :
                                                                  /-
                                                                    α : Type u_1
                                                                    β : Type u_3
                                                                    γ : Type u_5
                                                                    f : α → β → γ
                                                                    s : Set α
                                                                    t : Set β
                                                                    p : γ → Prop
                                                                    ⊢ Iff (Exists fun z => And (Membership.mem (Set.image2 f s t) z) (p z)) (Exist …
                                                                  -/
    (∃ z ∈ image2 f s t, p z) ↔ ∃ x ∈ s, ∃ y ∈ t, p (f x y) := by aesop
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[deprecated (since := "2024-11-23")] alias forall_image2_iff := forall_mem_image2


@[simp]
theorem image2_subset_iff {u : Set γ} : image2 f s t ⊆ u ↔ ∀ x ∈ s, ∀ y ∈ t, f x y ∈ u :=
  forall_mem_image2


theorem image2_subset_iff_left : image2 f s t ⊆ u ↔ ∀ a ∈ s, (fun b => f a b) '' t ⊆ u := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    u : Set γ
    ⊢ Iff (HasSubset.Subset (Set.image2 f s t) u) (∀ (a : α), Membership.mem s a → …
  -/
  simp_rw [image2_subset_iff, image_subset_iff, subset_def, mem_preimage]
  /-
    🎉 no goals
  -/


theorem image2_subset_iff_right : image2 f s t ⊆ u ↔ ∀ b ∈ t, (fun a => f a b) '' s ⊆ u := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    u : Set γ
    ⊢ Iff (HasSubset.Subset (Set.image2 f s t) u) (∀ (b : β), Membership.mem t b → …
  -/
  simp_rw [image2_subset_iff, image_subset_iff, subset_def, mem_preimage, @forall₂_swap α]
  /-
    🎉 no goals
  -/


lemma image_prod : (fun x : α × β ↦ f x.1 x.2) '' s ×ˢ t = image2 f s t :=
                 /-
                   α : Type u_1
                   β : Type u_3
                   γ : Type u_5
                   f : α → β → γ
                   s : Set α
                   t : Set β
                   x✝ : γ
                   ⊢ Iff (Membership.mem (Set.image (fun x => f x.1 x.2) (SProd.sprod s t)) x✝) ( …
                 -/
  ext fun _ ↦ by simp [and_assoc]
                 /-
                   🎉 no goals
                 -/


@[simp] lemma image_uncurry_prod (s : Set α) (t : Set β) : uncurry f '' s ×ˢ t = image2 f s t :=
  image_prod _


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_3
                                                                             s : Set α
                                                                             t : Set β
                                                                             ⊢ ∀ (x : Prod α β), Iff (Membership.mem (Set.image2 Prod.mk s t) x) (Membershi …
                                                                           -/
@[simp] lemma image2_mk_eq_prod : image2 Prod.mk s t = s ×ˢ t := ext <| by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

-- Porting note: Removing `simp` - LHS does not simplify

lemma image2_curry (f : α × β → γ) (s : Set α) (t : Set β) :
    image2 (fun a b ↦ f (a, b)) s t = f '' s ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : Prod α β → γ
    s : Set α
    t : Set β
    ⊢ Eq (Set.image2 (fun a b => f { fst := a, snd := b }) s t) (Set.image f (SPro …
  -/
  simp [← image_uncurry_prod, uncurry]
  /-
    🎉 no goals
  -/


theorem image2_swap (s : Set α) (t : Set β) : image2 f s t = image2 (fun a b => f b a) t s := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    ⊢ Eq (Set.image2 f s t) (Set.image2 (fun a b => f b a) t s)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    x✝ : γ
    ⊢ Iff (Membership.mem (Set.image2 f s t) x✝) (Membership.mem (Set.image2 (fun  …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  constructor <;> rintro ⟨a, ha, b, hb, rfl⟩ <;> exact ⟨b, hb, a, ha, rfl⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem image2_union_left : image2 f (s ∪ s') t = image2 f s t ∪ image2 f s' t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s s' : Set α
    t : Set β
    ⊢ Eq (Set.image2 f (Union.union s s') t) (Union.union (Set.image2 f s t) (Set. …
  -/
  simp_rw [← image_prod, union_prod, image_union]
  /-
    🎉 no goals
  -/


theorem image2_union_right : image2 f s (t ∪ t') = image2 f s t ∪ image2 f s t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t t' : Set β
    ⊢ Eq (Set.image2 f s (Union.union t t')) (Union.union (Set.image2 f s t) (Set. …
  -/
  rw [← image2_swap, image2_union_left, image2_swap f, image2_swap f]
  /-
    🎉 no goals
  -/


lemma image2_inter_left (hf : Injective2 f) :
    image2 f (s ∩ s') t = image2 f s t ∩ image2 f s' t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s s' : Set α
    t : Set β
    hf : Function.Injective2 f
    ⊢ Eq (Set.image2 f (Inter.inter s s') t) (Inter.inter (Set.image2 f s t) (Set. …
  -/
  simp_rw [← image_uncurry_prod, inter_prod, image_inter hf.uncurry]
  /-
    🎉 no goals
  -/


lemma image2_inter_right (hf : Injective2 f) :
    image2 f s (t ∩ t') = image2 f s t ∩ image2 f s t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t t' : Set β
    hf : Function.Injective2 f
    ⊢ Eq (Set.image2 f s (Inter.inter t t')) (Inter.inter (Set.image2 f s t) (Set. …
  -/
  simp_rw [← image_uncurry_prod, prod_inter, image_inter hf.uncurry]
  /-
    🎉 no goals
  -/


@[simp]
theorem image2_empty_left : image2 f ∅ t = ∅ :=
            /-
              α : Type u_1
              β : Type u_3
              γ : Type u_5
              f : α → β → γ
              t : Set β
              ⊢ ∀ (x : γ), Iff (Membership.mem (Set.image2 f EmptyCollection.emptyCollection …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


@[simp]
theorem image2_empty_right : image2 f s ∅ = ∅ :=
            /-
              α : Type u_1
              β : Type u_3
              γ : Type u_5
              f : α → β → γ
              s : Set α
              ⊢ ∀ (x : γ), Iff (Membership.mem (Set.image2 f s EmptyCollection.emptyCollecti …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


theorem Nonempty.image2 : s.Nonempty → t.Nonempty → (image2 f s t).Nonempty :=
  fun ⟨_, ha⟩ ⟨_, hb⟩ => ⟨_, mem_image2_of_mem ha hb⟩


@[simp]
theorem image2_nonempty_iff : (image2 f s t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  ⟨fun ⟨_, a, ha, b, hb, _⟩ => ⟨⟨a, ha⟩, b, hb⟩, fun h => h.1.image2 h.2⟩


theorem Nonempty.of_image2_left (h : (Set.image2 f s t).Nonempty) : s.Nonempty :=
  (image2_nonempty_iff.1 h).1


theorem Nonempty.of_image2_right (h : (Set.image2 f s t).Nonempty) : t.Nonempty :=
  (image2_nonempty_iff.1 h).2


@[simp]
theorem image2_eq_empty_iff : image2 f s t = ∅ ↔ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    ⊢ Iff (Eq (Set.image2 f s t) EmptyCollection.emptyCollection) (Or (Eq s EmptyC …
  -/
  rw [← not_nonempty_iff_eq_empty, image2_nonempty_iff, not_and_or]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    ⊢ Iff (Or (Not s.Nonempty) (Not t.Nonempty)) (Or (Eq s EmptyCollection.emptyCo …
  -/
  simp [not_nonempty_iff_eq_empty]
  /-
    🎉 no goals
  -/


theorem Subsingleton.image2 (hs : s.Subsingleton) (ht : t.Subsingleton) (f : α → β → γ) :
    (image2 f s t).Subsingleton := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    s : Set α
    t : Set β
    hs : s.Subsingleton
    ht : t.Subsingleton
    f : α → β → γ
    ⊢ (Set.image2 f s t).Subsingleton
  -/
  rw [← image_prod]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    s : Set α
    t : Set β
    hs : s.Subsingleton
    ht : t.Subsingleton
    f : α → β → γ
    ⊢ (Set.image (fun x => f x.1 x.2) (SProd.sprod s t)).Subsingleton
  -/
  apply (hs.prod ht).image
  /-
    🎉 no goals
  -/


theorem image2_inter_subset_left : image2 f (s ∩ s') t ⊆ image2 f s t ∩ image2 f s' t :=
  Monotone.map_inf_le (fun _ _ ↦ image2_subset_right) s s'


theorem image2_inter_subset_right : image2 f s (t ∩ t') ⊆ image2 f s t ∩ image2 f s t' :=
  Monotone.map_inf_le (fun _ _ ↦ image2_subset_left) t t'


@[simp]
theorem image2_singleton_left : image2 f {a} t = f a '' t :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    γ : Type u_5
                    f : α → β → γ
                    t : Set β
                    a : α
                    x : γ
                    ⊢ Iff (Membership.mem (Set.image2 f (Singleton.singleton a) t) x) (Membership. …
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem image2_singleton_right : image2 f s {b} = (fun a => f a b) '' s :=
                  /-
                    α : Type u_1
                    β : Type u_3
                    γ : Type u_5
                    f : α → β → γ
                    s : Set α
                    b : β
                    x : γ
                    ⊢ Iff (Membership.mem (Set.image2 f s (Singleton.singleton b)) x) (Membership. …
                  -/
  ext fun x => by simp
                  /-
                    🎉 no goals
                  -/


                                                            /-
                                                              α : Type u_1
                                                              β : Type u_3
                                                              γ : Type u_5
                                                              f : α → β → γ
                                                              a : α
                                                              b : β
                                                              ⊢ Eq (Set.image2 f (Singleton.singleton a) (Singleton.singleton b)) (Singleton …
                                                            -/
theorem image2_singleton : image2 f {a} {b} = {f a b} := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem image2_insert_left : image2 f (insert a s) t = (fun b => f a b) '' t ∪ image2 f s t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    a : α
    ⊢ Eq (Set.image2 f (Insert.insert a s) t) (Union.union (Set.image (fun b => f  …
  -/
  rw [insert_eq, image2_union_left, image2_singleton_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem image2_insert_right : image2 f s (insert b t) = (fun a => f a b) '' s ∪ image2 f s t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s : Set α
    t : Set β
    b : β
    ⊢ Eq (Set.image2 f s (Insert.insert b t)) (Union.union (Set.image (fun a => f  …
  -/
  rw [insert_eq, image2_union_right, image2_singleton_right]
  /-
    🎉 no goals
  -/


@[congr]
theorem image2_congr (h : ∀ a ∈ s, ∀ b ∈ t, f a b = f' a b) : image2 f s t = image2 f' s t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f f' : α → β → γ
    s : Set α
    t : Set β
    h : ∀ (a : α), Membership.mem s a → ∀ (b : β), Membership.mem t b → Eq (f a b) …
    ⊢ Eq (Set.image2 f s t) (Set.image2 f' s t)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f f' : α → β → γ
    s : Set α
    t : Set β
    h : ∀ (a : α), Membership.mem s a → ∀ (b : β), Membership.mem t b → Eq (f a b) …
    x✝ : γ
    ⊢ Iff (Membership.mem (Set.image2 f s t) x✝) (Membership.mem (Set.image2 f' s  …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  constructor <;> rintro ⟨a, ha, b, hb, rfl⟩ <;> exact ⟨a, ha, b, hb, by rw [h a ha b hb]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A common special case of `image2_congr` -/
theorem image2_congr' (h : ∀ a b, f a b = f' a b) : image2 f s t = image2 f' s t :=
  image2_congr fun a _ b _ => h a b


theorem image_image2 (f : α → β → γ) (g : γ → δ) :
    g '' image2 f s t = image2 (fun a b => g (f a b)) s t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    s : Set α
    t : Set β
    f : α → β → γ
    g : γ → δ
    ⊢ Eq (Set.image g (Set.image2 f s t)) (Set.image2 (fun a b => g (f a b)) s t)
  -/
  simp only [← image_prod, image_image]
  /-
    🎉 no goals
  -/


theorem image2_image_left (f : γ → β → δ) (g : α → γ) :
    image2 f (g '' s) t = image2 (fun a b => f (g a) b) s t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    s : Set α
    t : Set β
    f : γ → β → δ
    g : α → γ
    ⊢ Eq (Set.image2 f (Set.image g s) t) (Set.image2 (fun a b => f (g a) b) s t)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem image2_image_right (f : α → γ → δ) (g : β → γ) :
    image2 f s (g '' t) = image2 (fun a b => f a (g b)) s t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    s : Set α
    t : Set β
    f : α → γ → δ
    g : β → γ
    ⊢ Eq (Set.image2 f s (Set.image g t)) (Set.image2 (fun a b => f a (g b)) s t)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem image2_left (h : t.Nonempty) : image2 (fun x _ => x) s t = s := by
  /-
    α : Type u_1
    β : Type u_3
    s : Set α
    t : Set β
    h : t.Nonempty
    ⊢ Eq (Set.image2 (fun x x_1 => x) s t) s
  -/
  simp [nonempty_def.mp h, Set.ext_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem image2_right (h : s.Nonempty) : image2 (fun _ y => y) s t = t := by
  /-
    α : Type u_1
    β : Type u_3
    s : Set α
    t : Set β
    h : s.Nonempty
    ⊢ Eq (Set.image2 (fun x y => y) s t) t
  -/
  simp [nonempty_def.mp h, Set.ext_iff]
  /-
    🎉 no goals
  -/


lemma image2_range (f : α' → β' → γ) (g : α → α') (h : β → β') :
    image2 f (range g) (range h) = range fun x : α × β ↦ f (g x.1) (h x.2) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    f : α' → β' → γ
    g : α → α'
    h : β → β'
    ⊢ Eq (Set.image2 f (Set.range g) (Set.range h)) (Set.range fun x => f (g x.1)  …
  -/
  simp_rw [← image_univ, image2_image_left, image2_image_right, ← image_prod, univ_prod_univ]
  /-
    🎉 no goals
  -/


theorem image2_assoc {f : δ → γ → ε} {g : α → β → δ} {f' : α → ε' → ε} {g' : β → γ → ε'}
    (h_assoc : ∀ a b c, f (g a b) c = f' a (g' b c)) :
    image2 f (image2 g s t) u = image2 f' s (image2 g' t u) :=
                                     /-
                                       α : Type u_1
                                       β : Type u_3
                                       γ : Type u_5
                                       δ : Type u_7
                                       ε : Type u_9
                                       ε' : Type u_10
                                       s : Set α
                                       t : Set β
                                       u : Set γ
                                       f : δ → γ → ε
                                       g : α → β → δ
                                       f' : α → ε' → ε
                                       g' : β → γ → ε'
                                       h_assoc : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (f' a (g' b c))
                                       x✝ : Set ε
                                       ⊢ Iff (HasSubset.Subset (Set.image2 f (Set.image2 g s t) u) x✝) (HasSubset.Sub …
                                     -/
  eq_of_forall_subset_iff fun _ ↦ by simp only [image2_subset_iff, forall_mem_image2, h_assoc]
                                     /-
                                       🎉 no goals
                                     -/


theorem image2_comm {g : β → α → γ} (h_comm : ∀ a b, f a b = g b a) : image2 f s t = image2 g t s :=
                                  /-
                                    α : Type u_1
                                    β : Type u_3
                                    γ : Type u_5
                                    f : α → β → γ
                                    s : Set α
                                    t : Set β
                                    g : β → α → γ
                                    h_comm : ∀ (a : α) (b : β), Eq (f a b) (g b a)
                                    ⊢ Eq (Set.image2 (fun a b => f b a) t s) (Set.image2 g t s)
                                  -/
  (image2_swap _ _ _).trans <| by simp_rw [h_comm]
                                  /-
                                    🎉 no goals
                                  -/


theorem image2_left_comm {f : α → δ → ε} {g : β → γ → δ} {f' : α → γ → δ'} {g' : β → δ' → ε}
    (h_left_comm : ∀ a b c, f a (g b c) = g' b (f' a c)) :
    image2 f s (image2 g t u) = image2 g' t (image2 f' s u) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : α → δ → ε
    g : β → γ → δ
    f' : α → γ → δ'
    g' : β → δ' → ε
    h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' b (f' a c))
    ⊢ Eq (Set.image2 f s (Set.image2 g t u)) (Set.image2 g' t (Set.image2 f' s u))
  -/
  rw [image2_swap f', image2_swap f]
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : α → δ → ε
    g : β → γ → δ
    f' : α → γ → δ'
    g' : β → δ' → ε
    h_left_comm : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' b (f' a c))
    ⊢ Eq (Set.image2 (fun a b => f b a) (Set.image2 g t u) s) (Set.image2 g' t (Se …
  -/
  exact image2_assoc fun _ _ _ => h_left_comm _ _ _
  /-
    🎉 no goals
  -/


theorem image2_right_comm {f : δ → γ → ε} {g : α → β → δ} {f' : α → γ → δ'} {g' : δ' → β → ε}
    (h_right_comm : ∀ a b c, f (g a b) c = g' (f' a c) b) :
    image2 f (image2 g s t) u = image2 g' (image2 f' s u) t := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : δ → γ → ε
    g : α → β → δ
    f' : α → γ → δ'
    g' : δ' → β → ε
    h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f' a c) b)
    ⊢ Eq (Set.image2 f (Set.image2 g s t) u) (Set.image2 g' (Set.image2 f' s u) t)
  -/
  rw [image2_swap g, image2_swap g']
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    δ' : Type u_8
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : δ → γ → ε
    g : α → β → δ
    f' : α → γ → δ'
    g' : δ' → β → ε
    h_right_comm : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f' a c) b)
    ⊢ Eq (Set.image2 f (Set.image2 (fun a b => g b a) t s) u) (Set.image2 (fun a b …
  -/
  exact image2_assoc fun _ _ _ => h_right_comm _ _ _
  /-
    🎉 no goals
  -/


theorem image2_image2_image2_comm {f : ε → ζ → ν} {g : α → β → ε} {h : γ → δ → ζ} {f' : ε' → ζ' → ν}
    {g' : α → γ → ε'} {h' : β → δ → ζ'}
    (h_comm : ∀ a b c d, f (g a b) (h c d) = f' (g' a c) (h' b d)) :
    image2 f (image2 g s t) (image2 h u v) = image2 f' (image2 g' s u) (image2 h' t v) := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    ε' : Type u_10
    ζ : Type u_11
    ζ' : Type u_12
    ν : Type u_13
    s : Set α
    t : Set β
    u : Set γ
    v : Set δ
    f : ε → ζ → ν
    g : α → β → ε
    h : γ → δ → ζ
    f' : ε' → ζ' → ν
    g' : α → γ → ε'
    h' : β → δ → ζ'
    h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
    ⊢ Eq (Set.image2 f (Set.image2 g s t) (Set.image2 h u v)) (Set.image2 f' (Set. …
  -/
  ext; constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      ε : Type u_9
      ε' : Type u_10
      ζ : Type u_11
      ζ' : Type u_12
      ν : Type u_13
      s : Set α
      t : Set β
      u : Set γ
      v : Set δ
      f : ε → ζ → ν
      g : α → β → ε
      h : γ → δ → ζ
      f' : ε' → ζ' → ν
      g' : α → γ → ε'
      h' : β → δ → ζ'
      h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
      x✝ : ν
      ⊢ Membership.mem (Set.image2 f (Set.image2 g s t) (Set.image2 h u v)) x✝ → Mem …
    -/
  · rintro ⟨_, ⟨a, ha, b, hb, rfl⟩, _, ⟨c, hc, d, hd, rfl⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.in …
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      ε : Type u_9
      ε' : Type u_10
      ζ : Type u_11
      ζ' : Type u_12
      ν : Type u_13
      s : Set α
      t : Set β
      u : Set γ
      v : Set δ
      f : ε → ζ → ν
      g : α → β → ε
      h : γ → δ → ζ
      f' : ε' → ζ' → ν
      g' : α → γ → ε'
      h' : β → δ → ζ'
      h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
      a : α
      ha : Membership.mem s a
      b : β
      hb : Membership.mem t b
      c : γ
      hc : Membership.mem u c
      d : δ
      hd : Membership.mem v d
      ⊢ Membership.mem (Set.image2 f' (Set.image2 g' s u) (Set.image2 h' t v)) (f (g …
    -/
    exact ⟨_, ⟨a, ha, c, hc, rfl⟩, _, ⟨b, hb, d, hd, rfl⟩, (h_comm _ _ _ _).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      ε : Type u_9
      ε' : Type u_10
      ζ : Type u_11
      ζ' : Type u_12
      ν : Type u_13
      s : Set α
      t : Set β
      u : Set γ
      v : Set δ
      f : ε → ζ → ν
      g : α → β → ε
      h : γ → δ → ζ
      f' : ε' → ζ' → ν
      g' : α → γ → ε'
      h' : β → δ → ζ'
      h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
      x✝ : ν
      ⊢ Membership.mem (Set.image2 f' (Set.image2 g' s u) (Set.image2 h' t v)) x✝ →  …
    -/
  · rintro ⟨_, ⟨a, ha, c, hc, rfl⟩, _, ⟨b, hb, d, hd, rfl⟩, rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      β : Type u_3
      γ : Type u_5
      δ : Type u_7
      ε : Type u_9
      ε' : Type u_10
      ζ : Type u_11
      ζ' : Type u_12
      ν : Type u_13
      s : Set α
      t : Set β
      u : Set γ
      v : Set δ
      f : ε → ζ → ν
      g : α → β → ε
      h : γ → δ → ζ
      f' : ε' → ζ' → ν
      g' : α → γ → ε'
      h' : β → δ → ζ'
      h_comm : ∀ (a : α) (b : β) (c : γ) (d : δ), Eq (f (g a b) (h c d)) (f' (g' a c …
      a : α
      ha : Membership.mem s a
      c : γ
      hc : Membership.mem u c
      b : β
      hb : Membership.mem t b
      d : δ
      hd : Membership.mem v d
      ⊢ Membership.mem (Set.image2 f (Set.image2 g s t) (Set.image2 h u v)) (f' (g'  …
    -/
    exact ⟨_, ⟨a, ha, b, hb, rfl⟩, _, ⟨c, hc, d, hd, rfl⟩, h_comm _ _ _ _⟩
    /-
      🎉 no goals
    -/


theorem image_image2_distrib {g : γ → δ} {f' : α' → β' → δ} {g₁ : α → α'} {g₂ : β → β'}
    (h_distrib : ∀ a b, g (f a b) = f' (g₁ a) (g₂ b)) :
    (image2 f s t).image g = image2 f' (s.image g₁) (t.image g₂) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    f : α → β → γ
    s : Set α
    t : Set β
    g : γ → δ
    f' : α' → β' → δ
    g₁ : α → α'
    g₂ : β → β'
    h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ a) (g₂ b))
    ⊢ Eq (Set.image g (Set.image2 f s t)) (Set.image2 f' (Set.image g₁ s) (Set.ima …
  -/
  simp_rw [image_image2, image2_image_left, image2_image_right, h_distrib]
  /-
    🎉 no goals
  -/


/-- Symmetric statement to `Set.image2_image_left_comm`. -/
theorem image_image2_distrib_left {g : γ → δ} {f' : α' → β → δ} {g' : α → α'}
    (h_distrib : ∀ a b, g (f a b) = f' (g' a) b) :
    (image2 f s t).image g = image2 f' (s.image g') t :=
                                               /-
                                                 α : Type u_1
                                                 α' : Type u_2
                                                 β : Type u_3
                                                 γ : Type u_5
                                                 δ : Type u_7
                                                 f : α → β → γ
                                                 s : Set α
                                                 t : Set β
                                                 g : γ → δ
                                                 f' : α' → β → δ
                                                 g' : α → α'
                                                 h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' a) b)
                                                 ⊢ Eq (Set.image2 f' (Set.image g' s) (Set.image (fun b => b) t)) (Set.image2 f …
                                               -/
  (image_image2_distrib h_distrib).trans <| by rw [image_id']
                                               /-
                                                 🎉 no goals
                                               -/


/-- Symmetric statement to `Set.image_image2_right_comm`. -/
theorem image_image2_distrib_right {g : γ → δ} {f' : α → β' → δ} {g' : β → β'}
    (h_distrib : ∀ a b, g (f a b) = f' a (g' b)) :
    (image2 f s t).image g = image2 f' s (t.image g') :=
                                               /-
                                                 α : Type u_1
                                                 β : Type u_3
                                                 β' : Type u_4
                                                 γ : Type u_5
                                                 δ : Type u_7
                                                 f : α → β → γ
                                                 s : Set α
                                                 t : Set β
                                                 g : γ → δ
                                                 f' : α → β' → δ
                                                 g' : β → β'
                                                 h_distrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' a (g' b))
                                                 ⊢ Eq (Set.image2 f' (Set.image (fun a => a) s) (Set.image g' t)) (Set.image2 f …
                                               -/
  (image_image2_distrib h_distrib).trans <| by rw [image_id']
                                               /-
                                                 🎉 no goals
                                               -/


/-- Symmetric statement to `Set.image_image2_distrib_left`. -/
theorem image2_image_left_comm {f : α' → β → γ} {g : α → α'} {f' : α → β → δ} {g' : δ → γ}
    (h_left_comm : ∀ a b, f (g a) b = g' (f' a b)) :
    image2 f (s.image g) t = (image2 f' s t).image g' :=
  (image_image2_distrib_left fun a b => (h_left_comm a b).symm).symm


/-- Symmetric statement to `Set.image_image2_distrib_right`. -/
theorem image_image2_right_comm {f : α → β' → γ} {g : β → β'} {f' : α → β → δ} {g' : δ → γ}
    (h_right_comm : ∀ a b, f a (g b) = g' (f' a b)) :
    image2 f s (t.image g) = (image2 f' s t).image g' :=
  (image_image2_distrib_right fun a b => (h_right_comm a b).symm).symm


/-- The other direction does not hold because of the `s`-`s` cross terms on the RHS. -/
theorem image2_distrib_subset_left {f : α → δ → ε} {g : β → γ → δ} {f₁ : α → β → β'}
    {f₂ : α → γ → γ'} {g' : β' → γ' → ε} (h_distrib : ∀ a b c, f a (g b c) = g' (f₁ a b) (f₂ a c)) :
    image2 f s (image2 g t u) ⊆ image2 g' (image2 f₁ s t) (image2 f₂ s u) := by
  /-
    α : Type u_1
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    γ' : Type u_6
    δ : Type u_7
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : α → δ → ε
    g : β → γ → δ
    f₁ : α → β → β'
    f₂ : α → γ → γ'
    g' : β' → γ' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' (f₁ a b) (f₂ a c))
    ⊢ HasSubset.Subset (Set.image2 f s (Set.image2 g t u)) (Set.image2 g' (Set.ima …
  -/
  rintro _ ⟨a, ha, _, ⟨b, hb, c, hc, rfl⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    γ' : Type u_6
    δ : Type u_7
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : α → δ → ε
    g : β → γ → δ
    f₁ : α → β → β'
    f₂ : α → γ → γ'
    g' : β' → γ' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' (f₁ a b) (f₂ a c))
    a : α
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    c : γ
    hc : Membership.mem u c
    ⊢ Membership.mem (Set.image2 g' (Set.image2 f₁ s t) (Set.image2 f₂ s u)) (f a  …
  -/
  rw [h_distrib]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    γ' : Type u_6
    δ : Type u_7
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : α → δ → ε
    g : β → γ → δ
    f₁ : α → β → β'
    f₂ : α → γ → γ'
    g' : β' → γ' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f a (g b c)) (g' (f₁ a b) (f₂ a c))
    a : α
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    c : γ
    hc : Membership.mem u c
    ⊢ Membership.mem (Set.image2 g' (Set.image2 f₁ s t) (Set.image2 f₂ s u)) (g' ( …
  -/
  exact mem_image2_of_mem (mem_image2_of_mem ha hb) (mem_image2_of_mem ha hc)
  /-
    🎉 no goals
  -/


/-- The other direction does not hold because of the `u`-`u` cross terms on the RHS. -/
theorem image2_distrib_subset_right {f : δ → γ → ε} {g : α → β → δ} {f₁ : α → γ → α'}
    {f₂ : β → γ → β'} {g' : α' → β' → ε} (h_distrib : ∀ a b c, f (g a b) c = g' (f₁ a c) (f₂ b c)) :
    image2 f (image2 g s t) u ⊆ image2 g' (image2 f₁ s u) (image2 f₂ t u) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : δ → γ → ε
    g : α → β → δ
    f₁ : α → γ → α'
    f₂ : β → γ → β'
    g' : α' → β' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f₁ a c) (f₂ b c))
    ⊢ HasSubset.Subset (Set.image2 f (Set.image2 g s t) u) (Set.image2 g' (Set.ima …
  -/
  rintro _ ⟨_, ⟨a, ha, b, hb, rfl⟩, c, hc, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : δ → γ → ε
    g : α → β → δ
    f₁ : α → γ → α'
    f₂ : β → γ → β'
    g' : α' → β' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f₁ a c) (f₂ b c))
    a : α
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    c : γ
    hc : Membership.mem u c
    ⊢ Membership.mem (Set.image2 g' (Set.image2 f₁ s u) (Set.image2 f₂ t u)) (f (g …
  -/
  rw [h_distrib]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    ε : Type u_9
    s : Set α
    t : Set β
    u : Set γ
    f : δ → γ → ε
    g : α → β → δ
    f₁ : α → γ → α'
    f₂ : β → γ → β'
    g' : α' → β' → ε
    h_distrib : ∀ (a : α) (b : β) (c : γ), Eq (f (g a b) c) (g' (f₁ a c) (f₂ b c))
    a : α
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    c : γ
    hc : Membership.mem u c
    ⊢ Membership.mem (Set.image2 g' (Set.image2 f₁ s u) (Set.image2 f₂ t u)) (g' ( …
  -/
  exact mem_image2_of_mem (mem_image2_of_mem ha hc) (mem_image2_of_mem hb hc)
  /-
    🎉 no goals
  -/


theorem image_image2_antidistrib {g : γ → δ} {f' : β' → α' → δ} {g₁ : β → β'} {g₂ : α → α'}
    (h_antidistrib : ∀ a b, g (f a b) = f' (g₁ b) (g₂ a)) :
    (image2 f s t).image g = image2 f' (t.image g₁) (s.image g₂) := by
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    f : α → β → γ
    s : Set α
    t : Set β
    g : γ → δ
    f' : β' → α' → δ
    g₁ : β → β'
    g₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ b) (g₂ a))
    ⊢ Eq (Set.image g (Set.image2 f s t)) (Set.image2 f' (Set.image g₁ t) (Set.ima …
  -/
  rw [image2_swap f]
  /-
    α : Type u_1
    α' : Type u_2
    β : Type u_3
    β' : Type u_4
    γ : Type u_5
    δ : Type u_7
    f : α → β → γ
    s : Set α
    t : Set β
    g : γ → δ
    f' : β' → α' → δ
    g₁ : β → β'
    g₂ : α → α'
    h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g₁ b) (g₂ a))
    ⊢ Eq (Set.image g (Set.image2 (fun a b => f b a) t s)) (Set.image2 f' (Set.ima …
  -/
  exact image_image2_distrib fun _ _ => h_antidistrib _ _
  /-
    🎉 no goals
  -/


/-- Symmetric statement to `Set.image2_image_left_anticomm`. -/
theorem image_image2_antidistrib_left {g : γ → δ} {f' : β' → α → δ} {g' : β → β'}
    (h_antidistrib : ∀ a b, g (f a b) = f' (g' b) a) :
    (image2 f s t).image g = image2 f' (t.image g') s :=
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_3
                                                         β' : Type u_4
                                                         γ : Type u_5
                                                         δ : Type u_7
                                                         f : α → β → γ
                                                         s : Set α
                                                         t : Set β
                                                         g : γ → δ
                                                         f' : β' → α → δ
                                                         g' : β → β'
                                                         h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' (g' b) a)
                                                         ⊢ Eq (Set.image2 f' (Set.image g' t) (Set.image (fun a => a) s)) (Set.image2 f …
                                                       -/
  (image_image2_antidistrib h_antidistrib).trans <| by rw [image_id']
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Symmetric statement to `Set.image_image2_right_anticomm`. -/
theorem image_image2_antidistrib_right {g : γ → δ} {f' : β → α' → δ} {g' : α → α'}
    (h_antidistrib : ∀ a b, g (f a b) = f' b (g' a)) :
    (image2 f s t).image g = image2 f' t (s.image g') :=
                                                       /-
                                                         α : Type u_1
                                                         α' : Type u_2
                                                         β : Type u_3
                                                         γ : Type u_5
                                                         δ : Type u_7
                                                         f : α → β → γ
                                                         s : Set α
                                                         t : Set β
                                                         g : γ → δ
                                                         f' : β → α' → δ
                                                         g' : α → α'
                                                         h_antidistrib : ∀ (a : α) (b : β), Eq (g (f a b)) (f' b (g' a))
                                                         ⊢ Eq (Set.image2 f' (Set.image (fun b => b) t) (Set.image g' s)) (Set.image2 f …
                                                       -/
  (image_image2_antidistrib h_antidistrib).trans <| by rw [image_id']
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Symmetric statement to `Set.image_image2_antidistrib_left`. -/
theorem image2_image_left_anticomm {f : α' → β → γ} {g : α → α'} {f' : β → α → δ} {g' : δ → γ}
    (h_left_anticomm : ∀ a b, f (g a) b = g' (f' b a)) :
    image2 f (s.image g) t = (image2 f' t s).image g' :=
  (image_image2_antidistrib_left fun a b => (h_left_anticomm b a).symm).symm


/-- Symmetric statement to `Set.image_image2_antidistrib_right`. -/
theorem image_image2_right_anticomm {f : α → β' → γ} {g : β → β'} {f' : β → α → δ} {g' : δ → γ}
    (h_right_anticomm : ∀ a b, f a (g b) = g' (f' b a)) :
    image2 f s (t.image g) = (image2 f' t s).image g' :=
  (image_image2_antidistrib_right fun a b => (h_right_anticomm b a).symm).symm


/-- If `a` is a left identity for `f : α → β → β`, then `{a}` is a left identity for
`Set.image2 f`. -/
lemma image2_left_identity {f : α → β → β} {a : α} (h : ∀ b, f a b = b) (t : Set β) :
    image2 f {a} t = t := by
  /-
    α : Type u_1
    β : Type u_3
    f : α → β → β
    a : α
    h : ∀ (b : β), Eq (f a b) b
    t : Set β
    ⊢ Eq (Set.image2 f (Singleton.singleton a) t) t
  -/
  rw [image2_singleton_left, show f a = id from funext h, image_id]
  /-
    🎉 no goals
  -/


/-- If `b` is a right identity for `f : α → β → α`, then `{b}` is a right identity for
`Set.image2 f`. -/
lemma image2_right_identity {f : α → β → α} {b : β} (h : ∀ a, f a b = a) (s : Set α) :
    image2 f s {b} = s := by
  /-
    α : Type u_1
    β : Type u_3
    f : α → β → α
    b : β
    h : ∀ (a : α), Eq (f a b) a
    s : Set α
    ⊢ Eq (Set.image2 f s (Singleton.singleton b)) s
  -/
  rw [image2_singleton_right, funext h, image_id']
  /-
    🎉 no goals
  -/


theorem image2_inter_union_subset_union :
    image2 f (s ∩ s') (t ∪ t') ⊆ image2 f s t ∪ image2 f s' t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s s' : Set α
    t t' : Set β
    ⊢ HasSubset.Subset (Set.image2 f (Inter.inter s s') (Union.union t t')) (Union …
  -/
  rw [image2_union_right]
  exact
    union_subset_union (image2_subset_right inter_subset_left)
      (image2_subset_right inter_subset_right)


theorem image2_union_inter_subset_union :
    image2 f (s ∪ s') (t ∩ t') ⊆ image2 f s t ∪ image2 f s' t' := by
  /-
    α : Type u_1
    β : Type u_3
    γ : Type u_5
    f : α → β → γ
    s s' : Set α
    t t' : Set β
    ⊢ HasSubset.Subset (Set.image2 f (Union.union s s') (Inter.inter t t')) (Union …
  -/
  rw [image2_union_left]
  exact
    union_subset_union (image2_subset_left inter_subset_left)
      (image2_subset_left inter_subset_right)


theorem image2_inter_union_subset {f : α → α → β} {s t : Set α} (hf : ∀ a b, f a b = f b a) :
    image2 f (s ∩ t) (s ∪ t) ⊆ image2 f s t := by
  /-
    α : Type u_1
    β : Type u_3
    f : α → α → β
    s t : Set α
    hf : ∀ (a b : α), Eq (f a b) (f b a)
    ⊢ HasSubset.Subset (Set.image2 f (Inter.inter s t) (Union.union s t)) (Set.ima …
  -/
  rw [inter_comm]
  /-
    α : Type u_1
    β : Type u_3
    f : α → α → β
    s t : Set α
    hf : ∀ (a b : α), Eq (f a b) (f b a)
    ⊢ HasSubset.Subset (Set.image2 f (Inter.inter t s) (Union.union s t)) (Set.ima …
  -/
  exact image2_inter_union_subset_union.trans (union_subset (image2_comm hf).subset Subset.rfl)
  /-
    🎉 no goals
  -/


theorem image2_union_inter_subset {f : α → α → β} {s t : Set α} (hf : ∀ a b, f a b = f b a) :
    image2 f (s ∪ t) (s ∩ t) ⊆ image2 f s t := by
  /-
    α : Type u_1
    β : Type u_3
    f : α → α → β
    s t : Set α
    hf : ∀ (a b : α), Eq (f a b) (f b a)
    ⊢ HasSubset.Subset (Set.image2 f (Union.union s t) (Inter.inter s t)) (Set.ima …
  -/
  rw [image2_comm hf]
  /-
    α : Type u_1
    β : Type u_3
    f : α → α → β
    s t : Set α
    hf : ∀ (a b : α), Eq (f a b) (f b a)
    ⊢ HasSubset.Subset (Set.image2 f (Inter.inter s t) (Union.union s t)) (Set.ima …
  -/
  exact image2_inter_union_subset hf
  /-
    🎉 no goals
  -/


