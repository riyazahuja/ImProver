@[to_additive]
theorem image_list_prod (f : F) :
    ∀ l : List (Set α), (f : α → β) '' l.prod = (l.map fun s => f '' s).prod
  | [] => image_one.trans <| congr_arg singleton (map_one f)
                  /-
                    α : Type u_2
                    β : Type u_3
                    F : Type u_4
                    inst✝³ : FunLike F α β
                    inst✝² : Monoid α
                    inst✝¹ : Monoid β
                    inst✝ : MonoidHomClass F α β
                    f : F
                    a : Set α
                    as : List (Set α)
                    ⊢ Eq (Set.image (⇑f) (List.cons a as).prod) (List.map (fun s => Set.image (⇑f) …
                  -/
  | a :: as => by rw [List.map_cons, List.prod_cons, List.prod_cons, image_mul, image_list_prod _ _]
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem image_multiset_prod (f : F) :
    ∀ m : Multiset (Set α), (f : α → β) '' m.prod = (m.map fun s => f '' s).prod :=
  Quotient.ind <| by
    simpa only [Multiset.quot_mk_to_coe, Multiset.prod_coe, Multiset.map_coe] using
      image_list_prod f


@[to_additive]
theorem image_finset_prod (f : F) (m : Finset ι) (s : ι → Set α) :
    ((f : α → β) '' ∏ i ∈ m, s i) = ∏ i ∈ m, f '' s i :=
  (image_multiset_prod f _).trans <| congr_arg Multiset.prod <| Multiset.map_map _ _ _


/-- The n-ary version of `Set.mem_mul`. -/
@[to_additive " The n-ary version of `Set.mem_add`. "]
theorem mem_finset_prod (t : Finset ι) (f : ι → Set α) (a : α) :
    (a ∈ ∏ i ∈ t, f i) ↔ ∃ (g : ι → α) (_ : ∀ {i}, i ∈ t → g i ∈ f i), ∏ i ∈ t, g i = a := by
  classical
    induction' t using Finset.induction_on with i is hi ih generalizing a
    · simp_rw [Finset.prod_empty, Set.mem_one]
      exact ⟨fun h ↦ ⟨fun _ ↦ a, fun hi ↦ False.elim (Finset.not_mem_empty _ hi), h.symm⟩,
        fun ⟨_, _, hf⟩ ↦ hf.symm⟩
    rw [Finset.prod_insert hi, Set.mem_mul]
    simp_rw [Finset.prod_insert hi]
    simp_rw [ih]
    constructor
    · rintro ⟨x, y, hx, ⟨g, hg, rfl⟩, rfl⟩
      refine ⟨Function.update g i x, ?_, ?_⟩
      · intro j hj
        obtain rfl | hj := Finset.mem_insert.mp hj
        · rwa [Function.update_self]
        · rw [update_of_ne (ne_of_mem_of_not_mem hj hi)]
          exact hg hj
      · rw [Finset.prod_update_of_not_mem hi, Function.update_self]
    · rintro ⟨g, hg, rfl⟩
      exact ⟨g i, hg (is.mem_insert_self _), is.prod g,
        ⟨⟨g, fun hi ↦ hg (Finset.mem_insert_of_mem hi), rfl⟩, rfl⟩⟩


/-- A version of `Set.mem_finset_prod` with a simpler RHS for products over a Fintype. -/
@[to_additive " A version of `Set.mem_finset_sum` with a simpler RHS for sums over a Fintype. "]
theorem mem_fintype_prod [Fintype ι] (f : ι → Set α) (a : α) :
    (a ∈ ∏ i, f i) ↔ ∃ (g : ι → α) (_ : ∀ i, g i ∈ f i), ∏ i, g i = a := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : Fintype ι
    f : ι → Set α
    a : α
    ⊢ Iff (Membership.mem (Finset.univ.prod fun i => f i) a) (Exists fun g => Exis …
  -/
  rw [mem_finset_prod]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : Fintype ι
    f : ι → Set α
    a : α
    ⊢ Iff (Exists fun g => Exists fun x => Eq (Finset.univ.prod fun i => g i) a) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An n-ary version of `Set.mul_mem_mul`. -/
@[to_additive " An n-ary version of `Set.add_mem_add`. "]
theorem list_prod_mem_list_prod (t : List ι) (f : ι → Set α) (g : ι → α) (hg : ∀ i ∈ t, g i ∈ f i) :
    (t.map g).prod ∈ (t.map f).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    t : List ι
    f : ι → Set α
    g : ι → α
    hg : ∀ (i : ι), Membership.mem t i → Membership.mem (f i) (g i)
    ⊢ Membership.mem (List.map f t).prod (List.map g t).prod
  -/
  induction' t with h tl ih
    /-
      case nil
      ι : Type u_1
      α : Type u_2
      inst✝ : CommMonoid α
      f : ι → Set α
      g : ι → α
      hg : ∀ (i : ι), Membership.mem List.nil i → Membership.mem (f i) (g i)
      ⊢ Membership.mem (List.map f List.nil).prod (List.map g List.nil).prod
    -/
  · simp_rw [List.map_nil, List.prod_nil, Set.mem_one]
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      α : Type u_2
      inst✝ : CommMonoid α
      f : ι → Set α
      g : ι → α
      h : ι
      tl : List ι
      ih : (∀ (i : ι), Membership.mem tl i → Membership.mem (f i) (g i)) → Membershi …
      hg : ∀ (i : ι), Membership.mem (List.cons h tl) i → Membership.mem (f i) (g i)
      ⊢ Membership.mem (List.map f (List.cons h tl)).prod (List.map g (List.cons h t …
    -/
  · simp_rw [List.map_cons, List.prod_cons]
    exact mul_mem_mul (hg h <| List.mem_cons_self _ _)
      (ih fun i hi ↦ hg i <| List.mem_cons_of_mem _ hi)


/-- An n-ary version of `Set.mul_subset_mul`. -/
@[to_additive " An n-ary version of `Set.add_subset_add`. "]
theorem list_prod_subset_list_prod (t : List ι) (f₁ f₂ : ι → Set α) (hf : ∀ i ∈ t, f₁ i ⊆ f₂ i) :
    (t.map f₁).prod ⊆ (t.map f₂).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    t : List ι
    f₁ f₂ : ι → Set α
    hf : ∀ (i : ι), Membership.mem t i → HasSubset.Subset (f₁ i) (f₂ i)
    ⊢ HasSubset.Subset (List.map f₁ t).prod (List.map f₂ t).prod
  -/
  induction' t with h tl ih
    /-
      case nil
      ι : Type u_1
      α : Type u_2
      inst✝ : CommMonoid α
      f₁ f₂ : ι → Set α
      hf : ∀ (i : ι), Membership.mem List.nil i → HasSubset.Subset (f₁ i) (f₂ i)
      ⊢ HasSubset.Subset (List.map f₁ List.nil).prod (List.map f₂ List.nil).prod
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      α : Type u_2
      inst✝ : CommMonoid α
      f₁ f₂ : ι → Set α
      h : ι
      tl : List ι
      ih : (∀ (i : ι), Membership.mem tl i → HasSubset.Subset (f₁ i) (f₂ i)) → HasSu …
      hf : ∀ (i : ι), Membership.mem (List.cons h tl) i → HasSubset.Subset (f₁ i) (f …
      ⊢ HasSubset.Subset (List.map f₁ (List.cons h tl)).prod (List.map f₂ (List.cons …
    -/
  · simp_rw [List.map_cons, List.prod_cons]
    exact mul_subset_mul (hf h <| List.mem_cons_self _ _)
      (ih fun i hi ↦ hf i <| List.mem_cons_of_mem _ hi)


@[to_additive]
theorem list_prod_singleton {M : Type*} [CommMonoid M] (s : List M) :
    (s.map fun i ↦ ({i} : Set M)).prod = {s.prod} :=
  (map_list_prod (singletonMonoidHom : M →* Set M) _).symm


/-- An n-ary version of `Set.mul_mem_mul`. -/
@[to_additive " An n-ary version of `Set.add_mem_add`. "]
theorem multiset_prod_mem_multiset_prod (t : Multiset ι) (f : ι → Set α) (g : ι → α)
    (hg : ∀ i ∈ t, g i ∈ f i) : (t.map g).prod ∈ (t.map f).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    t : Multiset ι
    f : ι → Set α
    g : ι → α
    hg : ∀ (i : ι), Membership.mem t i → Membership.mem (f i) (g i)
    ⊢ Membership.mem (Multiset.map f t).prod (Multiset.map g t).prod
  -/
  induction t using Quotient.inductionOn
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    f : ι → Set α
    g : ι → α
    a✝ : List ι
    hg : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → Membersh …
    ⊢ Membership.mem (Multiset.map f (Quotient.mk (List.isSetoid ι) a✝)).prod (Mul …
  -/
  simp_rw [Multiset.quot_mk_to_coe, Multiset.map_coe, Multiset.prod_coe]
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    f : ι → Set α
    g : ι → α
    a✝ : List ι
    hg : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → Membersh …
    ⊢ Membership.mem (List.map f a✝).prod (List.map g a✝).prod
  -/
  exact list_prod_mem_list_prod _ _ _ hg
  /-
    🎉 no goals
  -/


/-- An n-ary version of `Set.mul_subset_mul`. -/
@[to_additive " An n-ary version of `Set.add_subset_add`. "]
theorem multiset_prod_subset_multiset_prod (t : Multiset ι) (f₁ f₂ : ι → Set α)
    (hf : ∀ i ∈ t, f₁ i ⊆ f₂ i) : (t.map f₁).prod ⊆ (t.map f₂).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    t : Multiset ι
    f₁ f₂ : ι → Set α
    hf : ∀ (i : ι), Membership.mem t i → HasSubset.Subset (f₁ i) (f₂ i)
    ⊢ HasSubset.Subset (Multiset.map f₁ t).prod (Multiset.map f₂ t).prod
  -/
  induction t using Quotient.inductionOn
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    f₁ f₂ : ι → Set α
    a✝ : List ι
    hf : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → HasSubse …
    ⊢ HasSubset.Subset (Multiset.map f₁ (Quotient.mk (List.isSetoid ι) a✝)).prod ( …
  -/
  simp_rw [Multiset.quot_mk_to_coe, Multiset.map_coe, Multiset.prod_coe]
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    f₁ f₂ : ι → Set α
    a✝ : List ι
    hf : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → HasSubse …
    ⊢ HasSubset.Subset (List.map f₁ a✝).prod (List.map f₂ a✝).prod
  -/
  exact list_prod_subset_list_prod _ _ _ hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multiset_prod_singleton {M : Type*} [CommMonoid M] (s : Multiset M) :
    (s.map fun i ↦ ({i} : Set M)).prod = {s.prod} :=
  (map_multiset_prod (singletonMonoidHom : M →* Set M) _).symm


/-- An n-ary version of `Set.mul_mem_mul`. -/
@[to_additive " An n-ary version of `Set.add_mem_add`. "]
theorem finset_prod_mem_finset_prod (t : Finset ι) (f : ι → Set α) (g : ι → α)
    (hg : ∀ i ∈ t, g i ∈ f i) : (∏ i ∈ t, g i) ∈ ∏ i ∈ t, f i :=
  multiset_prod_mem_multiset_prod _ _ _ hg


/-- An n-ary version of `Set.mul_subset_mul`. -/
@[to_additive " An n-ary version of `Set.add_subset_add`. "]
theorem finset_prod_subset_finset_prod (t : Finset ι) (f₁ f₂ : ι → Set α)
    (hf : ∀ i ∈ t, f₁ i ⊆ f₂ i) : ∏ i ∈ t, f₁ i ⊆ ∏ i ∈ t, f₂ i :=
  multiset_prod_subset_multiset_prod _ _ _ hf


@[to_additive]
theorem finset_prod_singleton {M ι : Type*} [CommMonoid M] (s : Finset ι) (I : ι → M) :
    ∏ i ∈ s, ({I i} : Set M) = {∏ i ∈ s, I i} :=
  (map_prod (singletonMonoidHom : M →* Set M) _ _).symm


/-- The n-ary version of `Set.image_mul_prod`. -/
@[to_additive "The n-ary version of `Set.add_image_prod`. "]
theorem image_finset_prod_pi (l : Finset ι) (S : ι → Set α) :
    (fun f : ι → α => ∏ i ∈ l, f i) '' (l : Set ι).pi S = ∏ i ∈ l, S i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    l : Finset ι
    S : ι → Set α
    ⊢ Eq (Set.image (fun f => l.prod fun i => f i) ((↑l).pi S)) (l.prod fun i => S …
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝ : CommMonoid α
    l : Finset ι
    S : ι → Set α
    x✝ : α
    ⊢ Iff (Membership.mem (Set.image (fun f => l.prod fun i => f i) ((↑l).pi S)) x …
  -/
  simp_rw [mem_finset_prod, mem_image, mem_pi, exists_prop, Finset.mem_coe]
  /-
    🎉 no goals
  -/


/-- A special case of `Set.image_finset_prod_pi` for `Finset.univ`. -/
@[to_additive "A special case of `Set.image_finset_sum_pi` for `Finset.univ`. "]
theorem image_fintype_prod_pi [Fintype ι] (S : ι → Set α) :
    (fun f : ι → α => ∏ i, f i) '' univ.pi S = ∏ i, S i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : Fintype ι
    S : ι → Set α
    ⊢ Eq (Set.image (fun f => Finset.univ.prod fun i => f i) (Set.univ.pi S)) (Fin …
  -/
  simpa only [Finset.coe_univ] using image_finset_prod_pi Finset.univ S
  /-
    🎉 no goals
  -/


