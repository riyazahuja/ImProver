/-- A **corner** of a set `A` in an abelian group is a triple of points of the form
`(x, y), (x + d, y), (x, y + d)`. It is **nontrivial** if `d ≠ 0`.

Here we define it as triples `(x₁, y₁), (x₂, y₁), (x₁, y₂)` where `x₁ + y₂ = x₂ + y₁` in order for
the definition to make sense in commutative monoids, the motivating example being `ℕ`. -/
@[mk_iff]
structure IsCorner (A : Set (G × G)) (x₁ y₁ x₂ y₂ : G) : Prop where
  fst_fst_mem : (x₁, y₁) ∈ A
  fst_snd_mem : (x₁, y₂) ∈ A
  snd_fst_mem : (x₂, y₁) ∈ A
  add_eq_add : x₁ + y₂ = x₂ + y₁


/-- A **corner-free set** in an abelian group is a set containing no non-trivial corner. -/
def IsCornerFree (A : Set (G × G)) : Prop := ∀ ⦃x₁ y₁ x₂ y₂⦄, IsCorner A x₁ y₁ x₂ y₂ → x₁ = x₂


/-- A convenient restatement of corner-freeness in terms of an ambient product set. -/
lemma isCornerFree_iff (hAs : A ⊆ s ×ˢ s) :
    IsCornerFree A ↔ ∀ ⦃x₁⦄, x₁ ∈ s → ∀ ⦃y₁⦄, y₁ ∈ s → ∀ ⦃x₂⦄, x₂ ∈ s → ∀ ⦃y₂⦄, y₂ ∈ s →
      IsCorner A x₁ y₁ x₂ y₂ → x₁ = x₂ where
  mp hA _x₁ _ _y₁ _ _x₂ _ _y₂ _ hxy := hA hxy
  mpr hA _x₁ _y₁ _x₂ _y₂ hxy := hA (hAs hxy.fst_fst_mem).1 (hAs hxy.fst_fst_mem).2
    (hAs hxy.snd_fst_mem).1 (hAs hxy.fst_snd_mem).2 hxy


lemma IsCorner.mono (hAB : A ⊆ B) (hA : IsCorner A x₁ y₁ x₂ y₂) : IsCorner B x₁ y₁ x₂ y₂ where
  fst_fst_mem := hAB hA.fst_fst_mem
  fst_snd_mem := hAB hA.fst_snd_mem
  snd_fst_mem := hAB hA.snd_fst_mem
  add_eq_add := hA.add_eq_add


lemma IsCornerFree.mono (hAB : A ⊆ B) (hB : IsCornerFree B) : IsCornerFree A :=
  fun _x₁ _y₁ _x₂ _y₂ hxyd ↦ hB <| hxyd.mono hAB


                                                                  /-
                                                                    G : Type u_1
                                                                    inst✝ : AddCommMonoid G
                                                                    x₁ y₁ x₂ y₂ : G
                                                                    ⊢ Not (IsCorner EmptyCollection.emptyCollection x₁ y₁ x₂ y₂)
                                                                  -/
@[simp] lemma not_isCorner_empty : ¬ IsCorner ∅ x₁ y₁ x₂ y₂ := by simp [isCorner_iff]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp] lemma Set.Subsingleton.isCornerFree (hA : A.Subsingleton) : IsCornerFree A :=
                                /-
                                  G : Type u_1
                                  inst✝ : AddCommMonoid G
                                  A : Set (Prod G G)
                                  hA : A.Subsingleton
                                  _x₁ _y₁ _x₂ _y₂ : G
                                  hxyd : IsCorner A _x₁ _y₁ _x₂ _y₂
                                  ⊢ Eq _x₁ _x₂
                                -/
  fun _x₁ _y₁ _x₂ _y₂ hxyd ↦ by simpa using hA hxyd.fst_fst_mem hxyd.snd_fst_mem
                                /-
                                  🎉 no goals
                                -/


lemma isCornerFree_empty : IsCornerFree (∅ : Set (G × G)) := subsingleton_empty.isCornerFree

lemma isCornerFree_singleton (x : G × G) : IsCornerFree {x} := subsingleton_singleton.isCornerFree


/-- Corners are preserved under `2`-Freiman homomorphisms. --/
lemma IsCorner.image (hf : IsAddFreimanHom 2 s t f) (hAs : (A : Set (G × G)) ⊆ s ×ˢ s)
    (hA : IsCorner A x₁ y₁ x₂ y₂) : IsCorner (Prod.map f f '' A) (f x₁) (f y₁) (f x₂) (f y₂) := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : AddCommMonoid G
    inst✝ : AddCommMonoid H
    A : Set (Prod G G)
    s : Set G
    t : Set H
    f : G → H
    x₁ y₁ x₂ y₂ : G
    hf : IsAddFreimanHom 2 s t f
    hAs : HasSubset.Subset A (SProd.sprod s s)
    hA : IsCorner A x₁ y₁ x₂ y₂
    ⊢ IsCorner (Set.image (Prod.map f f) A) (f x₁) (f y₁) (f x₂) (f y₂)
  -/
  obtain ⟨hx₁y₁, hx₁y₂, hx₂y₁, hxy⟩ := hA
  exact ⟨mem_image_of_mem _ hx₁y₁, mem_image_of_mem _ hx₁y₂, mem_image_of_mem _ hx₂y₁,
    hf.add_eq_add (hAs hx₁y₁).1 (hAs hx₁y₂).2 (hAs hx₂y₁).1 (hAs hx₁y₁).2 hxy⟩


/-- Corners are preserved under `2`-Freiman homomorphisms. --/
lemma IsCornerFree.of_image (hf : IsAddFreimanHom 2 s t f) (hf' : s.InjOn f)
    (hAs : (A : Set (G × G)) ⊆ s ×ˢ s) (hA : IsCornerFree (Prod.map f f '' A)) : IsCornerFree A :=
  fun _x₁ _y₁ _x₂ _y₂ hxy ↦
    hf' (hAs hxy.fst_fst_mem).1 (hAs hxy.snd_fst_mem).1 <| hA <| hxy.image hf hAs


lemma isCorner_image (hf : IsAddFreimanIso 2 s t f) (hAs : A ⊆ s ×ˢ s)
    (hx₁ : x₁ ∈ s) (hy₁ : y₁ ∈ s) (hx₂ : x₂ ∈ s) (hy₂ : y₂ ∈ s) :
    IsCorner (Prod.map f f '' A) (f x₁) (f y₁) (f x₂) (f y₂) ↔ IsCorner A x₁ y₁ x₂ y₂ := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : AddCommMonoid G
    inst✝ : AddCommMonoid H
    A : Set (Prod G G)
    s : Set G
    t : Set H
    f : G → H
    x₁ y₁ x₂ y₂ : G
    hf : IsAddFreimanIso 2 s t f
    hAs : HasSubset.Subset A (SProd.sprod s s)
    hx₁ : Membership.mem s x₁
    hy₁ : Membership.mem s y₁
    hx₂ : Membership.mem s x₂
    hy₂ : Membership.mem s y₂
    ⊢ Iff (IsCorner (Set.image (Prod.map f f) A) (f x₁) (f y₁) (f x₂) (f y₂)) (IsC …
  -/
  have hf' := hf.bijOn.injOn.prodMap hf.bijOn.injOn
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : AddCommMonoid G
    inst✝ : AddCommMonoid H
    A : Set (Prod G G)
    s : Set G
    t : Set H
    f : G → H
    x₁ y₁ x₂ y₂ : G
    hf : IsAddFreimanIso 2 s t f
    hAs : HasSubset.Subset A (SProd.sprod s s)
    hx₁ : Membership.mem s x₁
    hy₁ : Membership.mem s y₁
    hx₂ : Membership.mem s x₂
    hy₂ : Membership.mem s y₂
    hf' : Set.InjOn (fun x => { fst := f x.1, snd := f x.2 }) (SProd.sprod s s)
    ⊢ Iff (IsCorner (Set.image (Prod.map f f) A) (f x₁) (f y₁) (f x₂) (f y₂)) (IsC …
  -/
  rw [isCorner_iff, isCorner_iff]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : AddCommMonoid G
    inst✝ : AddCommMonoid H
    A : Set (Prod G G)
    s : Set G
    t : Set H
    f : G → H
    x₁ y₁ x₂ y₂ : G
    hf : IsAddFreimanIso 2 s t f
    hAs : HasSubset.Subset A (SProd.sprod s s)
    hx₁ : Membership.mem s x₁
    hy₁ : Membership.mem s y₁
    hx₂ : Membership.mem s x₂
    hy₂ : Membership.mem s y₂
    hf' : Set.InjOn (fun x => { fst := f x.1, snd := f x.2 }) (SProd.sprod s s)
    ⊢ Iff (And (Membership.mem (Set.image (Prod.map f f) A) { fst := f x₁, snd :=  …
  -/
  congr!
    /-
      case a.h.e'_1.a
      G : Type u_1
      H : Type u_2
      inst✝¹ : AddCommMonoid G
      inst✝ : AddCommMonoid H
      A : Set (Prod G G)
      s : Set G
      t : Set H
      f : G → H
      x₁ y₁ x₂ y₂ : G
      hf : IsAddFreimanIso 2 s t f
      hAs : HasSubset.Subset A (SProd.sprod s s)
      hx₁ : Membership.mem s x₁
      hy₁ : Membership.mem s y₁
      hx₂ : Membership.mem s x₂
      hy₂ : Membership.mem s y₂
      hf' : Set.InjOn (fun x => { fst := f x.1, snd := f x.2 }) (SProd.sprod s s)
      ⊢ Iff (Membership.mem (Set.image (Prod.map f f) A) { fst := f x₁, snd := f y₁  …
    -/
  · exact hf'.mem_image_iff hAs (mk_mem_prod hx₁ hy₁)
    /-
      🎉 no goals
    -/
    /-
      case a.h.e'_2.h.e'_1.a
      G : Type u_1
      H : Type u_2
      inst✝¹ : AddCommMonoid G
      inst✝ : AddCommMonoid H
      A : Set (Prod G G)
      s : Set G
      t : Set H
      f : G → H
      x₁ y₁ x₂ y₂ : G
      hf : IsAddFreimanIso 2 s t f
      hAs : HasSubset.Subset A (SProd.sprod s s)
      hx₁ : Membership.mem s x₁
      hy₁ : Membership.mem s y₁
      hx₂ : Membership.mem s x₂
      hy₂ : Membership.mem s y₂
      hf' : Set.InjOn (fun x => { fst := f x.1, snd := f x.2 }) (SProd.sprod s s)
      ⊢ Iff (Membership.mem (Set.image (Prod.map f f) A) { fst := f x₁, snd := f y₂  …
    -/
  · exact hf'.mem_image_iff hAs (mk_mem_prod hx₁ hy₂)
    /-
      🎉 no goals
    -/
    /-
      case a.h.e'_2.h.e'_2.h.e'_1.a
      G : Type u_1
      H : Type u_2
      inst✝¹ : AddCommMonoid G
      inst✝ : AddCommMonoid H
      A : Set (Prod G G)
      s : Set G
      t : Set H
      f : G → H
      x₁ y₁ x₂ y₂ : G
      hf : IsAddFreimanIso 2 s t f
      hAs : HasSubset.Subset A (SProd.sprod s s)
      hx₁ : Membership.mem s x₁
      hy₁ : Membership.mem s y₁
      hx₂ : Membership.mem s x₂
      hy₂ : Membership.mem s y₂
      hf' : Set.InjOn (fun x => { fst := f x.1, snd := f x.2 }) (SProd.sprod s s)
      ⊢ Iff (Membership.mem (Set.image (Prod.map f f) A) { fst := f x₂, snd := f y₁  …
    -/
  · exact hf'.mem_image_iff hAs (mk_mem_prod hx₂ hy₁)
    /-
      🎉 no goals
    -/
    /-
      case a.h.e'_2.h.e'_2.h.e'_2.a
      G : Type u_1
      H : Type u_2
      inst✝¹ : AddCommMonoid G
      inst✝ : AddCommMonoid H
      A : Set (Prod G G)
      s : Set G
      t : Set H
      f : G → H
      x₁ y₁ x₂ y₂ : G
      hf : IsAddFreimanIso 2 s t f
      hAs : HasSubset.Subset A (SProd.sprod s s)
      hx₁ : Membership.mem s x₁
      hy₁ : Membership.mem s y₁
      hx₂ : Membership.mem s x₂
      hy₂ : Membership.mem s y₂
      hf' : Set.InjOn (fun x => { fst := f x.1, snd := f x.2 }) (SProd.sprod s s)
      ⊢ Iff (Eq (HAdd.hAdd (f x₁) (f y₂)) (HAdd.hAdd (f x₂) (f y₁))) (Eq (HAdd.hAdd  …
    -/
  · exact hf.add_eq_add hx₁ hy₂ hx₂ hy₁
    /-
      🎉 no goals
    -/


lemma isCornerFree_image (hf : IsAddFreimanIso 2 s t f) (hAs : A ⊆ s ×ˢ s) :
    IsCornerFree (Prod.map f f '' A) ↔ IsCornerFree A := by
  have : Prod.map f f '' A ⊆ t ×ˢ t :=
    ((hf.bijOn.mapsTo.prodMap hf.bijOn.mapsTo).mono hAs Subset.rfl).image_subset
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : AddCommMonoid G
    inst✝ : AddCommMonoid H
    A : Set (Prod G G)
    s : Set G
    t : Set H
    f : G → H
    hf : IsAddFreimanIso 2 s t f
    hAs : HasSubset.Subset A (SProd.sprod s s)
    this : HasSubset.Subset (Set.image (Prod.map f f) A) (SProd.sprod t t)
    ⊢ Iff (IsCornerFree (Set.image (Prod.map f f) A)) (IsCornerFree A)
  -/
  rw [isCornerFree_iff hAs, isCornerFree_iff this]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : AddCommMonoid G
    inst✝ : AddCommMonoid H
    A : Set (Prod G G)
    s : Set G
    t : Set H
    f : G → H
    hf : IsAddFreimanIso 2 s t f
    hAs : HasSubset.Subset A (SProd.sprod s s)
    this : HasSubset.Subset (Set.image (Prod.map f f) A) (SProd.sprod t t)
    ⊢ Iff (∀ ⦃x₁ : H⦄, Membership.mem t x₁ → ∀ ⦃y₁ : H⦄, Membership.mem t y₁ → ∀ ⦃ …
  -/
  simp +contextual only [hf.bijOn.forall, isCorner_image hf hAs, hf.bijOn.injOn.eq_iff]
  /-
    🎉 no goals
  -/


alias ⟨IsCorner.of_image, _⟩ := isCorner_image

alias ⟨_, IsCornerFree.image⟩ := isCornerFree_image


