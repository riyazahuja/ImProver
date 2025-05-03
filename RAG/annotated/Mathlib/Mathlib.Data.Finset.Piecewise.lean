/-- `s.piecewise f g` is the function equal to `f` on the finset `s`, and to `g` on its
complement. -/
def piecewise [∀ j, Decidable (j ∈ s)] : ∀ i, π i := fun i ↦ if i ∈ s then f i else g i


lemma piecewise_insert_self [DecidableEq ι] {j : ι} [∀ i, Decidable (i ∈ insert j s)] :
                                             /-
                                               ι : Type u_1
                                               π : ι → Sort u_2
                                               s : Finset ι
                                               f g : (i : ι) → π i
                                               inst✝¹ : DecidableEq ι
                                               j : ι
                                               inst✝ : (i : ι) → Decidable (Membership.mem (Insert.insert j s) i)
                                               ⊢ Eq ((Insert.insert j s).piecewise f g j) (f j)
                                             -/
    (insert j s).piecewise f g j = f j := by simp [piecewise]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
lemma piecewise_empty [∀ i : ι, Decidable (i ∈ (∅ : Finset ι))] : piecewise ∅ f g = g := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    f g : (i : ι) → π i
    inst✝ : (i : ι) → Decidable (Membership.mem EmptyCollection.emptyCollection i)
    ⊢ Eq (EmptyCollection.emptyCollection.piecewise f g) g
  -/
  ext i
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    f g : (i : ι) → π i
    inst✝ : (i : ι) → Decidable (Membership.mem EmptyCollection.emptyCollection i)
    i : ι
    ⊢ Eq (EmptyCollection.emptyCollection.piecewise f g i) (g i)
  -/
  simp [piecewise]
  /-
    🎉 no goals
  -/


@[norm_cast move]
lemma piecewise_coe [∀ j, Decidable (j ∈ (s : Set ι))] :
    (s : Set ι).piecewise f g = s.piecewise f g := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : (j : ι) → Decidable (Membership.mem (↑s) j)
    ⊢ Eq ((↑s).piecewise f g) (s.piecewise f g)
  -/
  ext
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : (j : ι) → Decidable (Membership.mem (↑s) j)
    x✝ : ι
    ⊢ Eq ((↑s).piecewise f g x✝) (s.piecewise f g x✝)
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
lemma piecewise_eq_of_mem {i : ι} (hi : i ∈ s) : s.piecewise f g i = f i := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝ : (j : ι) → Decidable (Membership.mem s j)
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (s.piecewise f g i) (f i)
  -/
  simp [piecewise, hi]
  /-
    🎉 no goals
  -/


@[simp]
lemma piecewise_eq_of_not_mem {i : ι} (hi : i ∉ s) : s.piecewise f g i = g i := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝ : (j : ι) → Decidable (Membership.mem s j)
    i : ι
    hi : Not (Membership.mem s i)
    ⊢ Eq (s.piecewise f g i) (g i)
  -/
  simp [piecewise, hi]
  /-
    🎉 no goals
  -/


lemma piecewise_congr {f f' g g' : ∀ i, π i} (hf : ∀ i ∈ s, f i = f' i)
    (hg : ∀ i ∉ s, g i = g' i) : s.piecewise f g = s.piecewise f' g' :=
  funext fun i => if_ctx_congr Iff.rfl (hf i) (hg i)


@[simp]
lemma piecewise_insert_of_ne [DecidableEq ι] {i j : ι} [∀ i, Decidable (i ∈ insert j s)]
                                                                         /-
                                                                           ι : Type u_1
                                                                           π : ι → Sort u_2
                                                                           s : Finset ι
                                                                           f g : (i : ι) → π i
                                                                           inst✝² : (j : ι) → Decidable (Membership.mem s j)
                                                                           inst✝¹ : DecidableEq ι
                                                                           i j : ι
                                                                           inst✝ : (i : ι) → Decidable (Membership.mem (Insert.insert j s) i)
                                                                           h : Ne i j
                                                                           ⊢ Eq ((Insert.insert j s).piecewise f g i) (s.piecewise f g i)
                                                                         -/
    (h : i ≠ j) : (insert j s).piecewise f g i = s.piecewise f g i := by simp [piecewise, h]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma piecewise_insert [DecidableEq ι] (j : ι) [∀ i, Decidable (i ∈ insert j s)] :
    (insert j s).piecewise f g = update (s.piecewise f g) j (f j) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝² : (j : ι) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq ι
    j : ι
    inst✝ : (i : ι) → Decidable (Membership.mem (Insert.insert j s) i)
    ⊢ Eq ((Insert.insert j s).piecewise f g) (Function.update (s.piecewise f g) j  …
  -/
  classical simp only [← piecewise_coe, coe_insert, ← Set.piecewise_insert]
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝² : (j : ι) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq ι
    j : ι
    inst✝ : (i : ι) → Decidable (Membership.mem (Insert.insert j s) i)
    ⊢ Eq ((↑(Insert.insert j s)).piecewise f g) ((Insert.insert j ↑s).piecewise f g)
  -/
  ext
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝² : (j : ι) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq ι
    j : ι
    inst✝ : (i : ι) → Decidable (Membership.mem (Insert.insert j s) i)
    x✝ : ι
    ⊢ Eq ((↑(Insert.insert j s)).piecewise f g x✝) ((Insert.insert j ↑s).piecewise …
  -/
  congr
  /-
    case h.e_s
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝² : (j : ι) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq ι
    j : ι
    inst✝ : (i : ι) → Decidable (Membership.mem (Insert.insert j s) i)
    x✝ : ι
    ⊢ Eq (↑(Insert.insert j s)) (Insert.insert j ↑s)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma piecewise_cases {i} (p : π i → Prop) (hf : p (f i)) (hg : p (g i)) :
    p (s.piecewise f g i) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝ : (j : ι) → Decidable (Membership.mem s j)
    i : ι
    p : π i → Prop
    hf : p (f i)
    hg : p (g i)
    ⊢ p (s.piecewise f g i)
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hi : i ∈ s <;> simpa [hi]
                          /-
                            🎉 no goals
                          -/


lemma piecewise_singleton [DecidableEq ι] (i : ι) : piecewise {i} f g = update g i (f i) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    f g : (i : ι) → π i
    inst✝ : DecidableEq ι
    i : ι
    ⊢ Eq ((Singleton.singleton i).piecewise f g) (Function.update g i (f i))
  -/
  rw [← insert_emptyc_eq, piecewise_insert, piecewise_empty]
  /-
    🎉 no goals
  -/


lemma piecewise_piecewise_of_subset_left {s t : Finset ι} [∀ i, Decidable (i ∈ s)]
    [∀ i, Decidable (i ∈ t)] (h : s ⊆ t) (f₁ f₂ g : ∀ a, π a) :
    s.piecewise (t.piecewise f₁ f₂) g = s.piecewise f₁ g :=
  s.piecewise_congr (fun _i hi => piecewise_eq_of_mem _ _ _ (h hi)) fun _ _ => rfl


@[simp]
lemma piecewise_idem_left (f₁ f₂ g : ∀ a, π a) :
    s.piecewise (s.piecewise f₁ f₂) g = s.piecewise f₁ g :=
  piecewise_piecewise_of_subset_left (Subset.refl _) _ _ _


lemma piecewise_piecewise_of_subset_right {s t : Finset ι} [∀ i, Decidable (i ∈ s)]
    [∀ i, Decidable (i ∈ t)] (h : t ⊆ s) (f g₁ g₂ : ∀ a, π a) :
    s.piecewise f (t.piecewise g₁ g₂) = s.piecewise f g₂ :=
  s.piecewise_congr (fun _ _ => rfl) fun _i hi => t.piecewise_eq_of_not_mem _ _ (mt (@h _) hi)


@[simp]
lemma piecewise_idem_right (f g₁ g₂ : ∀ a, π a) :
    s.piecewise f (s.piecewise g₁ g₂) = s.piecewise f g₂ :=
  piecewise_piecewise_of_subset_right (Subset.refl _) f g₁ g₂


lemma update_eq_piecewise {β : Type*} [DecidableEq ι] (f : ι → β) (i : ι) (v : β) :
    update f i v = piecewise (singleton i) (fun _ => v) f :=
  (piecewise_singleton (fun _ => v) _ _).symm


lemma update_piecewise [DecidableEq ι] (i : ι) (v : π i) :
    update (s.piecewise f g) i v = s.piecewise (update f i v) (update g i v) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    v : π i
    ⊢ Eq (Function.update (s.piecewise f g) i v) (s.piecewise (Function.update f i …
  -/
  ext j
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    v : π i
    j : ι
    ⊢ Eq (Function.update (s.piecewise f g) i v j) (s.piecewise (Function.update f …
  -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  rcases em (j = i) with (rfl | hj) <;> by_cases hs : j ∈ s <;> simp [*]
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma update_piecewise_of_mem [DecidableEq ι] {i : ι} (hi : i ∈ s) (v : π i) :
    update (s.piecewise f g) i v = s.piecewise (update f i v) g := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    hi : Membership.mem s i
    v : π i
    ⊢ Eq (Function.update (s.piecewise f g) i v) (s.piecewise (Function.update f i …
  -/
  rw [update_piecewise]
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    hi : Membership.mem s i
    v : π i
    ⊢ Eq (s.piecewise (Function.update f i v) (Function.update g i v)) (s.piecewis …
  -/
  refine s.piecewise_congr (fun _ _ => rfl) fun j hj => update_of_ne ?_ ..
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    hi : Membership.mem s i
    v : π i
    j : ι
    hj : Not (Membership.mem s j)
    ⊢ Ne j i
  -/
  exact fun h => hj (h.symm ▸ hi)
  /-
    🎉 no goals
  -/


lemma update_piecewise_of_not_mem [DecidableEq ι] {i : ι} (hi : i ∉ s) (v : π i) :
    update (s.piecewise f g) i v = s.piecewise f (update g i v) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    hi : Not (Membership.mem s i)
    v : π i
    ⊢ Eq (Function.update (s.piecewise f g) i v) (s.piecewise f (Function.update g …
  -/
  rw [update_piecewise]
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    hi : Not (Membership.mem s i)
    v : π i
    ⊢ Eq (s.piecewise (Function.update f i v) (Function.update g i v)) (s.piecewis …
  -/
  refine s.piecewise_congr (fun j hj => update_of_ne ?_ ..) fun _ _ => rfl
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f g : (i : ι) → π i
    inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
    inst✝ : DecidableEq ι
    i : ι
    hi : Not (Membership.mem s i)
    v : π i
    j : ι
    hj : Membership.mem s j
    ⊢ Ne j i
  -/
  exact fun h => hi (h ▸ hj)
  /-
    🎉 no goals
  -/


lemma piecewise_same : s.piecewise f f = f := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f : (i : ι) → π i
    inst✝ : (j : ι) → Decidable (Membership.mem s j)
    ⊢ Eq (s.piecewise f f) f
  -/
  ext i
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    s : Finset ι
    f : (i : ι) → π i
    inst✝ : (j : ι) → Decidable (Membership.mem s j)
    i : ι
    ⊢ Eq (s.piecewise f f i) (f i)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i ∈ s <;> simp [h]
                         /-
                           🎉 no goals
                         -/


@[simp]
lemma piecewise_univ [∀ i, Decidable (i ∈ (univ : Finset ι))] (f g : ∀ i, π i) :
    univ.piecewise f g = f := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → Decidable (Membership.mem Finset.univ i)
    f g : (i : ι) → π i
    ⊢ Eq (Finset.univ.piecewise f g) f
  -/
  ext i
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → Decidable (Membership.mem Finset.univ i)
    f g : (i : ι) → π i
    i : ι
    ⊢ Eq (Finset.univ.piecewise f g i) (f i)
  -/
  simp [piecewise]
  /-
    🎉 no goals
  -/


lemma piecewise_compl [DecidableEq ι] (s : Finset ι) [∀ i, Decidable (i ∈ s)]
    [∀ i, Decidable (i ∈ sᶜ)] (f g : ∀ i, π i) :
    sᶜ.piecewise f g = s.piecewise g f := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    s : Finset ι
    inst✝¹ : (i : ι) → Decidable (Membership.mem s i)
    inst✝ : (i : ι) → Decidable (Membership.mem (HasCompl.compl s) i)
    f g : (i : ι) → π i
    ⊢ Eq ((HasCompl.compl s).piecewise f g) (s.piecewise g f)
  -/
  ext i
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    s : Finset ι
    inst✝¹ : (i : ι) → Decidable (Membership.mem s i)
    inst✝ : (i : ι) → Decidable (Membership.mem (HasCompl.compl s) i)
    f g : (i : ι) → π i
    i : ι
    ⊢ Eq ((HasCompl.compl s).piecewise f g i) (s.piecewise g f i)
  -/
  simp [piecewise]
  /-
    🎉 no goals
  -/


@[simp]
lemma piecewise_erase_univ [DecidableEq ι] (i : ι) (f g : ∀ i, π i) :
    (Finset.univ.erase i).piecewise f g = Function.update f i (g i) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    i : ι
    f g : (i : ι) → π i
    ⊢ Eq ((Finset.univ.erase i).piecewise f g) (Function.update f i (g i))
  -/
  rw [← compl_singleton, piecewise_compl, piecewise_singleton]
  /-
    🎉 no goals
  -/


lemma piecewise_mem_set_pi (hf : f ∈ Set.pi t t') (hg : g ∈ Set.pi t t') :
    s.piecewise f g ∈ Set.pi t t' := by
  /-
    ι : Type u_1
    s : Finset ι
    inst✝ : (j : ι) → Decidable (Membership.mem s j)
    π : ι → Type u_3
    t : Set ι
    t' : (i : ι) → Set (π i)
    f g : (i : ι) → π i
    hf : Membership.mem (t.pi t') f
    hg : Membership.mem (t.pi t') g
    ⊢ Membership.mem (t.pi t') (s.piecewise f g)
  -/
  classical rw [← piecewise_coe]; exact Set.piecewise_mem_pi (↑s) hf hg
  /-
    🎉 no goals
  -/


lemma piecewise_le_of_le_of_le (hf : f ≤ h) (hg : g ≤ h) : s.piecewise f g ≤ h := fun x =>
  piecewise_cases s f g (· ≤ h x) (hf x) (hg x)


lemma le_piecewise_of_le_of_le (hf : h ≤ f) (hg : h ≤ g) : h ≤ s.piecewise f g := fun x =>
  piecewise_cases s f g (fun y => h x ≤ y) (hf x) (hg x)


lemma piecewise_le_piecewise' (hf : ∀ x ∈ s, f x ≤ f' x) (hg : ∀ x ∉ s, g x ≤ g' x) :
                                                       /-
                                                         ι : Type u_1
                                                         s : Finset ι
                                                         inst✝¹ : (j : ι) → Decidable (Membership.mem s j)
                                                         π : ι → Type u_3
                                                         f g f' g' : (i : ι) → π i
                                                         inst✝ : (i : ι) → Preorder (π i)
                                                         hf : ∀ (x : ι), Membership.mem s x → LE.le (f x) (f' x)
                                                         hg : ∀ (x : ι), Not (Membership.mem s x) → LE.le (g x) (g' x)
                                                         x : ι
                                                         ⊢ LE.le (s.piecewise f g x) (s.piecewise f' g' x)
                                                       -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    s.piecewise f g ≤ s.piecewise f' g' := fun x => by by_cases hx : x ∈ s <;> simp [hx, *]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma piecewise_le_piecewise (hf : f ≤ f') (hg : g ≤ g') : s.piecewise f g ≤ s.piecewise f' g' :=
  s.piecewise_le_piecewise' (fun x _ => hf x) fun x _ => hg x


lemma piecewise_mem_Icc_of_mem_of_mem (hf : f ∈ Set.Icc f' g') (hg : g ∈ Set.Icc f' g') :
    s.piecewise f g ∈ Set.Icc f' g' :=
  ⟨le_piecewise_of_le_of_le _ hf.1 hg.1, piecewise_le_of_le_of_le _ hf.2 hg.2⟩


lemma piecewise_mem_Icc (h : f ≤ g) : s.piecewise f g ∈ Set.Icc f g :=
  piecewise_mem_Icc_of_mem_of_mem _ (Set.left_mem_Icc.2 h) (Set.right_mem_Icc.2 h)


lemma piecewise_mem_Icc' (h : g ≤ f) : s.piecewise f g ∈ Set.Icc g f :=
  piecewise_mem_Icc_of_mem_of_mem _ (Set.right_mem_Icc.2 h) (Set.left_mem_Icc.2 h)


