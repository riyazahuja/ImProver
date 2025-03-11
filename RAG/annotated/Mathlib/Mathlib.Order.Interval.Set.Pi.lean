@[simp]
theorem pi_univ_Ici : (pi univ fun i ↦ Ici (x i)) = Ici x :=
                 /-
                   ι : Type u_1
                   α : ι → Type u_2
                   inst✝ : (i : ι) → Preorder (α i)
                   x y : (i : ι) → α i
                   ⊢ Iff (Membership.mem (Set.univ.pi fun i => Set.Ici (x i)) y) (Membership.mem  …
                 -/
  ext fun y ↦ by simp [Pi.le_def]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem pi_univ_Iic : (pi univ fun i ↦ Iic (x i)) = Iic x :=
                 /-
                   ι : Type u_1
                   α : ι → Type u_2
                   inst✝ : (i : ι) → Preorder (α i)
                   x y : (i : ι) → α i
                   ⊢ Iff (Membership.mem (Set.univ.pi fun i => Set.Iic (x i)) y) (Membership.mem  …
                 -/
  ext fun y ↦ by simp [Pi.le_def]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem pi_univ_Icc : (pi univ fun i ↦ Icc (x i) (y i)) = Icc x y :=
                 /-
                   ι : Type u_1
                   α : ι → Type u_2
                   inst✝ : (i : ι) → Preorder (α i)
                   x y✝ y : (i : ι) → α i
                   ⊢ Iff (Membership.mem (Set.univ.pi fun i => Set.Icc (x i) (y✝ i)) y) (Membersh …
                 -/
  ext fun y ↦ by simp [Pi.le_def, forall_and]
                 /-
                   🎉 no goals
                 -/


theorem piecewise_mem_Icc {s : Set ι} [∀ j, Decidable (j ∈ s)] {f₁ f₂ g₁ g₂ : ∀ i, α i}
    (h₁ : ∀ i ∈ s, f₁ i ∈ Icc (g₁ i) (g₂ i)) (h₂ : ∀ i ∉ s, f₂ i ∈ Icc (g₁ i) (g₂ i)) :
    s.piecewise f₁ f₂ ∈ Icc g₁ g₂ :=
  ⟨le_piecewise (fun i hi ↦ (h₁ i hi).1) fun i hi ↦ (h₂ i hi).1,
    piecewise_le (fun i hi ↦ (h₁ i hi).2) fun i hi ↦ (h₂ i hi).2⟩


theorem piecewise_mem_Icc' {s : Set ι} [∀ j, Decidable (j ∈ s)] {f₁ f₂ g₁ g₂ : ∀ i, α i}
    (h₁ : f₁ ∈ Icc g₁ g₂) (h₂ : f₂ ∈ Icc g₁ g₂) : s.piecewise f₁ f₂ ∈ Icc g₁ g₂ :=
  piecewise_mem_Icc (fun _ _ ↦ ⟨h₁.1 _, h₁.2 _⟩) fun _ _ ↦ ⟨h₂.1 _, h₂.2 _⟩


theorem pi_univ_Ioi_subset [Nonempty ι]: (pi univ fun i ↦ Ioi (x i)) ⊆ Ioi x := fun _ hz ↦
  ⟨fun i ↦ le_of_lt <| hz i trivial, fun h ↦
    (‹Nonempty ι›.elim) fun i ↦ not_lt_of_le (h i) (hz i trivial)⟩


theorem pi_univ_Iio_subset [Nonempty ι]: (pi univ fun i ↦ Iio (x i)) ⊆ Iio x :=
  pi_univ_Ioi_subset (α := fun i ↦ (α i)ᵒᵈ) x


theorem pi_univ_Ioo_subset [Nonempty ι]: (pi univ fun i ↦ Ioo (x i) (y i)) ⊆ Ioo x y := fun _ hx ↦
  ⟨(pi_univ_Ioi_subset _) fun i hi ↦ (hx i hi).1, (pi_univ_Iio_subset _) fun i hi ↦ (hx i hi).2⟩


theorem pi_univ_Ioc_subset [Nonempty ι]: (pi univ fun i ↦ Ioc (x i) (y i)) ⊆ Ioc x y := fun _ hx ↦
  ⟨(pi_univ_Ioi_subset _) fun i hi ↦ (hx i hi).1, fun i ↦ (hx i trivial).2⟩


theorem pi_univ_Ico_subset [Nonempty ι]: (pi univ fun i ↦ Ico (x i) (y i)) ⊆ Ico x y := fun _ hx ↦
  ⟨fun i ↦ (hx i trivial).1, (pi_univ_Iio_subset _) fun i hi ↦ (hx i hi).2⟩


theorem pi_univ_Ioc_update_left {x y : ∀ i, α i} {i₀ : ι} {m : α i₀} (hm : x i₀ ≤ m) :
    (pi univ fun i ↦ Ioc (update x i₀ m i) (y i)) =
      { z | m < z i₀ } ∩ pi univ fun i ↦ Ioc (x i) (y i) := by
  have : Ioc m (y i₀) = Ioi m ∩ Ioc (x i₀) (y i₀) := by
    rw [← Ioi_inter_Iic, ← Ioi_inter_Iic, ← inter_assoc,
      inter_eq_self_of_subset_left (Ioi_subset_Ioi hm)]
  simp_rw [univ_pi_update i₀ _ _ fun i z ↦ Ioc z (y i), ← pi_inter_compl ({i₀} : Set ι),
    singleton_pi', ← inter_assoc, this]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Preorder (α i)
    inst✝ : DecidableEq ι
    x y : (i : ι) → α i
    i₀ : ι
    m : α i₀
    hm : LE.le (x i₀) m
    this : Eq (Set.Ioc m (y i₀)) (Inter.inter (Set.Ioi m) (Set.Ioc (x i₀) (y i₀)))
    ⊢ Eq (Inter.inter (setOf fun x_1 => Membership.mem (Inter.inter (Set.Ioi m) (S …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem pi_univ_Ioc_update_right {x y : ∀ i, α i} {i₀ : ι} {m : α i₀} (hm : m ≤ y i₀) :
    (pi univ fun i ↦ Ioc (x i) (update y i₀ m i)) =
      { z | z i₀ ≤ m } ∩ pi univ fun i ↦ Ioc (x i) (y i) := by
  have : Ioc (x i₀) m = Iic m ∩ Ioc (x i₀) (y i₀) := by
    rw [← Ioi_inter_Iic, ← Ioi_inter_Iic, inter_left_comm,
      inter_eq_self_of_subset_left (Iic_subset_Iic.2 hm)]
  simp_rw [univ_pi_update i₀ y m fun i z ↦ Ioc (x i) z, ← pi_inter_compl ({i₀} : Set ι),
    singleton_pi', ← inter_assoc, this]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Preorder (α i)
    inst✝ : DecidableEq ι
    x y : (i : ι) → α i
    i₀ : ι
    m : α i₀
    hm : LE.le m (y i₀)
    this : Eq (Set.Ioc (x i₀) m) (Inter.inter (Set.Iic m) (Set.Ioc (x i₀) (y i₀)))
    ⊢ Eq (Inter.inter (setOf fun x_1 => Membership.mem (Inter.inter (Set.Iic m) (S …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem disjoint_pi_univ_Ioc_update_left_right {x y : ∀ i, α i} {i₀ : ι} {m : α i₀} :
    Disjoint (pi univ fun i ↦ Ioc (x i) (update y i₀ m i))
    (pi univ fun i ↦ Ioc (update x i₀ m i) (y i)) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Preorder (α i)
    inst✝ : DecidableEq ι
    x y : (i : ι) → α i
    i₀ : ι
    m : α i₀
    ⊢ Disjoint (Set.univ.pi fun i => Set.Ioc (x i) (Function.update y i₀ m i)) (Se …
  -/
  rw [disjoint_left]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Preorder (α i)
    inst✝ : DecidableEq ι
    x y : (i : ι) → α i
    i₀ : ι
    m : α i₀
    ⊢ ∀ ⦃a : (i : ι) → α i⦄, Membership.mem (Set.univ.pi fun i => Set.Ioc (x i) (F …
  -/
  rintro z h₁ h₂
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Preorder (α i)
    inst✝ : DecidableEq ι
    x y : (i : ι) → α i
    i₀ : ι
    m : α i₀
    z : (i : ι) → α i
    h₁ : Membership.mem (Set.univ.pi fun i => Set.Ioc (x i) (Function.update y i₀  …
    h₂ : Membership.mem (Set.univ.pi fun i => Set.Ioc (Function.update x i₀ m i) ( …
    ⊢ False
  -/
  refine (h₁ i₀ (mem_univ _)).2.not_lt ?_
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Preorder (α i)
    inst✝ : DecidableEq ι
    x y : (i : ι) → α i
    i₀ : ι
    m : α i₀
    z : (i : ι) → α i
    h₁ : Membership.mem (Set.univ.pi fun i => Set.Ioc (x i) (Function.update y i₀  …
    h₂ : Membership.mem (Set.univ.pi fun i => Set.Ioc (Function.update x i₀ m i) ( …
    ⊢ LT.lt (Function.update y i₀ m i₀) (z i₀)
  -/
  simpa only [Function.update_self] using (h₂ i₀ (mem_univ _)).1
  /-
    🎉 no goals
  -/


theorem image_update_Icc (f : ∀ i, α i) (i : ι) (a b : α i) :
    update f i '' Icc a b = Icc (update f i a) (update f i b) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → PartialOrder (α i)
    f : (i : ι) → α i
    i : ι
    a b : α i
    ⊢ Eq (Set.image (Function.update f i) (Set.Icc a b)) (Set.Icc (Function.update …
  -/
  ext x
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → PartialOrder (α i)
    f : (i : ι) → α i
    i : ι
    a b : α i
    x : (a : ι) → α a
    ⊢ Iff (Membership.mem (Set.image (Function.update f i) (Set.Icc a b)) x) (Memb …
  -/
  rw [← Set.pi_univ_Icc]
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → PartialOrder (α i)
    f : (i : ι) → α i
    i : ι
    a b : α i
    x : (a : ι) → α a
    ⊢ Iff (Membership.mem (Set.image (Function.update f i) (Set.Icc a b)) x) (Memb …
  -/
  refine ⟨?_, fun h => ⟨x i, ?_, ?_⟩⟩
    /-
      case h.refine_1
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → PartialOrder (α i)
      f : (i : ι) → α i
      i : ι
      a b : α i
      x : (a : ι) → α a
      ⊢ Membership.mem (Set.image (Function.update f i) (Set.Icc a b)) x → Membershi …
    -/
  · rintro ⟨c, hc, rfl⟩
    /-
      case h.refine_1.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → PartialOrder (α i)
      f : (i : ι) → α i
      i : ι
      a b c : α i
      hc : Membership.mem (Set.Icc a b) c
      ⊢ Membership.mem (Set.univ.pi fun i_1 => Set.Icc (Function.update f i a i_1) ( …
    -/
    simpa [update_le_update_iff]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → PartialOrder (α i)
      f : (i : ι) → α i
      i : ι
      a b : α i
      x : (a : ι) → α a
      h : Membership.mem (Set.univ.pi fun i_1 => Set.Icc (Function.update f i a i_1) …
      ⊢ Membership.mem (Set.Icc a b) (x i)
    -/
  · simpa only [Function.update_self] using h i (mem_univ i)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → PartialOrder (α i)
      f : (i : ι) → α i
      i : ι
      a b : α i
      x : (a : ι) → α a
      h : Membership.mem (Set.univ.pi fun i_1 => Set.Icc (Function.update f i a i_1) …
      ⊢ Eq (Function.update f i (x i)) x
    -/
  · ext j
    /-
      case h.refine_3.h
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → PartialOrder (α i)
      f : (i : ι) → α i
      i : ι
      a b : α i
      x : (a : ι) → α a
      h : Membership.mem (Set.univ.pi fun i_1 => Set.Icc (Function.update f i a i_1) …
      j : ι
      ⊢ Eq (Function.update f i (x i) j) (x j)
    -/
    obtain rfl | hij := eq_or_ne i j
      /-
        case h.refine_3.h.inl
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → PartialOrder (α i)
        f : (i : ι) → α i
        i : ι
        a b : α i
        x : (a : ι) → α a
        h : Membership.mem (Set.univ.pi fun i_1 => Set.Icc (Function.update f i a i_1) …
        ⊢ Eq (Function.update f i (x i) i) (x i)
      -/
    · exact Function.update_self ..
      /-
        🎉 no goals
      -/
      /-
        case h.refine_3.h.inr
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → PartialOrder (α i)
        f : (i : ι) → α i
        i : ι
        a b : α i
        x : (a : ι) → α a
        h : Membership.mem (Set.univ.pi fun i_1 => Set.Icc (Function.update f i a i_1) …
        j : ι
        hij : Ne i j
        ⊢ Eq (Function.update f i (x i) j) (x j)
      -/
    · simpa only [Function.update_of_ne hij.symm, le_antisymm_iff] using h j (mem_univ j)
      /-
        🎉 no goals
      -/


theorem image_update_Ico (f : ∀ i, α i) (i : ι) (a b : α i) :
    update f i '' Ico a b = Ico (update f i a) (update f i b) := by
  rw [← Icc_diff_right, ← Icc_diff_right, image_diff (update_injective _ _), image_singleton,
    image_update_Icc]


theorem image_update_Ioc (f : ∀ i, α i) (i : ι) (a b : α i) :
    update f i '' Ioc a b = Ioc (update f i a) (update f i b) := by
  rw [← Icc_diff_left, ← Icc_diff_left, image_diff (update_injective _ _), image_singleton,
    image_update_Icc]


theorem image_update_Ioo (f : ∀ i, α i) (i : ι) (a b : α i) :
    update f i '' Ioo a b = Ioo (update f i a) (update f i b) := by
  rw [← Ico_diff_left, ← Ico_diff_left, image_diff (update_injective _ _), image_singleton,
    image_update_Ico]


theorem image_update_Icc_left (f : ∀ i, α i) (i : ι) (a : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             a : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Icc a (f i))) (Set.Icc (Function.up …
                                                           -/
    update f i '' Icc a (f i) = Icc (update f i a) f := by simpa using image_update_Icc f i a (f i)
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Ico_left (f : ∀ i, α i) (i : ι) (a : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             a : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Ico a (f i))) (Set.Ico (Function.up …
                                                           -/
    update f i '' Ico a (f i) = Ico (update f i a) f := by simpa using image_update_Ico f i a (f i)
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Ioc_left (f : ∀ i, α i) (i : ι) (a : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             a : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Ioc a (f i))) (Set.Ioc (Function.up …
                                                           -/
    update f i '' Ioc a (f i) = Ioc (update f i a) f := by simpa using image_update_Ioc f i a (f i)
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Ioo_left (f : ∀ i, α i) (i : ι) (a : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             a : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Ioo a (f i))) (Set.Ioo (Function.up …
                                                           -/
    update f i '' Ioo a (f i) = Ioo (update f i a) f := by simpa using image_update_Ioo f i a (f i)
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Icc_right (f : ∀ i, α i) (i : ι) (b : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             b : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Icc (f i) b)) (Set.Icc f (Function. …
                                                           -/
    update f i '' Icc (f i) b = Icc f (update f i b) := by simpa using image_update_Icc f i (f i) b
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Ico_right (f : ∀ i, α i) (i : ι) (b : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             b : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Ico (f i) b)) (Set.Ico f (Function. …
                                                           -/
    update f i '' Ico (f i) b = Ico f (update f i b) := by simpa using image_update_Ico f i (f i) b
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Ioc_right (f : ∀ i, α i) (i : ι) (b : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             b : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Ioc (f i) b)) (Set.Ioc f (Function. …
                                                           -/
    update f i '' Ioc (f i) b = Ioc f (update f i b) := by simpa using image_update_Ioc f i (f i) b
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_update_Ioo_right (f : ∀ i, α i) (i : ι) (b : α i) :
                                                           /-
                                                             ι : Type u_1
                                                             α : ι → Type u_2
                                                             inst✝¹ : DecidableEq ι
                                                             inst✝ : (i : ι) → PartialOrder (α i)
                                                             f : (i : ι) → α i
                                                             i : ι
                                                             b : α i
                                                             ⊢ Eq (Set.image (Function.update f i) (Set.Ioo (f i) b)) (Set.Ioo f (Function. …
                                                           -/
    update f i '' Ioo (f i) b = Ioo f (update f i b) := by simpa using image_update_Ioo f i (f i) b
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
theorem image_mulSingle_Icc (i : ι) (a b : α i) :
    Pi.mulSingle i '' Icc a b = Icc (Pi.mulSingle i a) (Pi.mulSingle i b) :=
  image_update_Icc _ _ _ _


@[to_additive]
theorem image_mulSingle_Ico (i : ι) (a b : α i) :
    Pi.mulSingle i '' Ico a b = Ico (Pi.mulSingle i a) (Pi.mulSingle i b) :=
  image_update_Ico _ _ _ _


@[to_additive]
theorem image_mulSingle_Ioc (i : ι) (a b : α i) :
    Pi.mulSingle i '' Ioc a b = Ioc (Pi.mulSingle i a) (Pi.mulSingle i b) :=
  image_update_Ioc _ _ _ _


@[to_additive]
theorem image_mulSingle_Ioo (i : ι) (a b : α i) :
    Pi.mulSingle i '' Ioo a b = Ioo (Pi.mulSingle i a) (Pi.mulSingle i b) :=
  image_update_Ioo _ _ _ _


@[to_additive]
theorem image_mulSingle_Icc_left (i : ι) (a : α i) :
    Pi.mulSingle i '' Icc a 1 = Icc (Pi.mulSingle i a) 1 :=
  image_update_Icc_left _ _ _


@[to_additive]
theorem image_mulSingle_Ico_left (i : ι) (a : α i) :
    Pi.mulSingle i '' Ico a 1 = Ico (Pi.mulSingle i a) 1 :=
  image_update_Ico_left _ _ _


@[to_additive]
theorem image_mulSingle_Ioc_left (i : ι) (a : α i) :
    Pi.mulSingle i '' Ioc a 1 = Ioc (Pi.mulSingle i a) 1 :=
  image_update_Ioc_left _ _ _


@[to_additive]
theorem image_mulSingle_Ioo_left (i : ι) (a : α i) :
    Pi.mulSingle i '' Ioo a 1 = Ioo (Pi.mulSingle i a) 1 :=
  image_update_Ioo_left _ _ _


@[to_additive]
theorem image_mulSingle_Icc_right (i : ι) (b : α i) :
    Pi.mulSingle i '' Icc 1 b = Icc 1 (Pi.mulSingle i b) :=
  image_update_Icc_right _ _ _


@[to_additive]
theorem image_mulSingle_Ico_right (i : ι) (b : α i) :
    Pi.mulSingle i '' Ico 1 b = Ico 1 (Pi.mulSingle i b) :=
  image_update_Ico_right _ _ _


@[to_additive]
theorem image_mulSingle_Ioc_right (i : ι) (b : α i) :
    Pi.mulSingle i '' Ioc 1 b = Ioc 1 (Pi.mulSingle i b) :=
  image_update_Ioc_right _ _ _


@[to_additive]
theorem image_mulSingle_Ioo_right (i : ι) (b : α i) :
    Pi.mulSingle i '' Ioo 1 b = Ioo 1 (Pi.mulSingle i b) :=
  image_update_Ioo_right _ _ _


@[simp]
theorem pi_univ_uIcc (a b : ∀ i, α i) : (pi univ fun i => uIcc (a i) (b i)) = uIcc a b :=
  pi_univ_Icc _ _


theorem image_update_uIcc (f : ∀ i, α i) (i : ι) (a b : α i) :
    update f i '' uIcc a b = uIcc (update f i a) (update f i b) :=
                                         /-
                                           ι : Type u_1
                                           α : ι → Type u_2
                                           inst✝¹ : (i : ι) → Lattice (α i)
                                           inst✝ : DecidableEq ι
                                           f : (i : ι) → α i
                                           i : ι
                                           a b : α i
                                           ⊢ Eq (Set.Icc (Function.update f i (Min.min a b)) (Function.update f i (Max.ma …
                                         -/
  (image_update_Icc _ _ _ _).trans <| by simp_rw [uIcc, update_sup, update_inf]
                                         /-
                                           🎉 no goals
                                         -/


theorem image_update_uIcc_left (f : ∀ i, α i) (i : ι) (a : α i) :
    update f i '' uIcc a (f i) = uIcc (update f i a) f := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Lattice (α i)
    inst✝ : DecidableEq ι
    f : (i : ι) → α i
    i : ι
    a : α i
    ⊢ Eq (Set.image (Function.update f i) (Set.uIcc a (f i))) (Set.uIcc (Function. …
  -/
  simpa using image_update_uIcc f i a (f i)
  /-
    🎉 no goals
  -/


theorem image_update_uIcc_right (f : ∀ i, α i) (i : ι) (b : α i) :
    update f i '' uIcc (f i) b = uIcc f (update f i b) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Lattice (α i)
    inst✝ : DecidableEq ι
    f : (i : ι) → α i
    i : ι
    b : α i
    ⊢ Eq (Set.image (Function.update f i) (Set.uIcc (f i) b)) (Set.uIcc f (Functio …
  -/
  simpa using image_update_uIcc f i (f i) b
  /-
    🎉 no goals
  -/


@[to_additive]
theorem image_mulSingle_uIcc (i : ι) (a b : α i) :
    Pi.mulSingle i '' uIcc a b = uIcc (Pi.mulSingle i a) (Pi.mulSingle i b) :=
  image_update_uIcc _ _ _ _


@[to_additive]
theorem image_mulSingle_uIcc_left (i : ι) (a : α i) :
    Pi.mulSingle i '' uIcc a 1 = uIcc (Pi.mulSingle i a) 1 :=
  image_update_uIcc_left _ _ _


@[to_additive]
theorem image_mulSingle_uIcc_right (i : ι) (b : α i) :
    Pi.mulSingle i '' uIcc 1 b = uIcc 1 (Pi.mulSingle i b) :=
  image_update_uIcc_right _ _ _


theorem pi_univ_Ioc_update_union (x y : ∀ i, α i) (i₀ : ι) (m : α i₀) (hm : m ∈ Icc (x i₀) (y i₀)) :
    ((pi univ fun i ↦ Ioc (x i) (update y i₀ m i)) ∪
        pi univ fun i ↦ Ioc (update x i₀ m i) (y i)) =
      pi univ fun i ↦ Ioc (x i) (y i) := by
  simp_rw [pi_univ_Ioc_update_left hm.1, pi_univ_Ioc_update_right hm.2, ← union_inter_distrib_right,
    ← setOf_or, le_or_lt, setOf_true, univ_inter]


/-- If `x`, `y`, `x'`, and `y'` are functions `Π i : ι, α i`, then
the set difference between the box `[x, y]` and the product of the open intervals `(x' i, y' i)`
is covered by the union of the following boxes: for each `i : ι`, we take
`[x, update y i (x' i)]` and `[update x i (y' i), y]`.

E.g., if `x' = x` and `y' = y`, then this lemma states that the difference between a closed box
`[x, y]` and the corresponding open box `{z | ∀ i, x i < z i < y i}` is covered by the union
of the faces of `[x, y]`. -/
theorem Icc_diff_pi_univ_Ioo_subset (x y x' y' : ∀ i, α i) :
    (Icc x y \ pi univ fun i ↦ Ioo (x' i) (y' i)) ⊆
    (⋃ i : ι, Icc x (update y i (x' i))) ∪ ⋃ i : ι, Icc (update x i (y' i)) y := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y x' y' : (i : ι) → α i
    ⊢ HasSubset.Subset (SDiff.sdiff (Set.Icc x y) (Set.univ.pi fun i => Set.Ioo (x …
  -/
  rintro a ⟨⟨hxa, hay⟩, ha'⟩
  /-
    case intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y x' y' a : (i : ι) → α i
    ha' : Not (Membership.mem (Set.univ.pi fun i => Set.Ioo (x' i) (y' i)) a)
    hxa : LE.le x a
    hay : LE.le a y
    ⊢ Membership.mem (Union.union (Set.iUnion fun i => Set.Icc x (Function.update  …
  -/
  simp only [mem_pi, mem_univ, mem_Ioo, true_implies, not_forall] at ha'
  simp only [le_update_iff, update_le_iff, mem_union, mem_iUnion, mem_Icc,
    hxa, hay _, hxa _, hay, ← exists_or]
  /-
    case intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y x' y' a : (i : ι) → α i
    hxa : LE.le x a
    hay : LE.le a y
    ha' : Exists fun x => Not (And (LT.lt (x' x) (a x)) (LT.lt (a x) (y' x)))
    ⊢ Exists fun x => Or (And True (And (LE.le (a x) (x' x)) (∀ (j : ι), Ne j x →  …
  -/
  rcases ha' with ⟨w, hw⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y x' y' a : (i : ι) → α i
    hxa : LE.le x a
    hay : LE.le a y
    w : ι
    hw : Not (And (LT.lt (x' w) (a w)) (LT.lt (a w) (y' w)))
    ⊢ Exists fun x => Or (And True (And (LE.le (a x) (x' x)) (∀ (j : ι), Ne j x →  …
  -/
  apply Exists.intro w
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y x' y' a : (i : ι) → α i
    hxa : LE.le x a
    hay : LE.le a y
    w : ι
    hw : Not (And (LT.lt (x' w) (a w)) (LT.lt (a w) (y' w)))
    ⊢ Or (And True (And (LE.le (a w) (x' w)) (∀ (j : ι), Ne j w → True))) (And (An …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases lt_or_le (x' w) (a w) <;> simp_all
                                  /-
                                    🎉 no goals
                                  -/


/-- If `x`, `y`, `z` are functions `Π i : ι, α i`, then
the set difference between the box `[x, z]` and the product of the intervals `(y i, z i]`
is covered by the union of the boxes `[x, update z i (y i)]`.

E.g., if `x = y`, then this lemma states that the difference between a closed box
`[x, y]` and the product of half-open intervals `{z | ∀ i, x i < z i ≤ y i}` is covered by the union
of the faces of `[x, y]` adjacent to `x`. -/
theorem Icc_diff_pi_univ_Ioc_subset (x y z : ∀ i, α i) :
    (Icc x z \ pi univ fun i ↦ Ioc (y i) (z i)) ⊆ ⋃ i : ι, Icc x (update z i (y i)) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y z : (i : ι) → α i
    ⊢ HasSubset.Subset (SDiff.sdiff (Set.Icc x z) (Set.univ.pi fun i => Set.Ioc (y …
  -/
  rintro a ⟨⟨hax, haz⟩, hay⟩
  /-
    case intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → LinearOrder (α i)
    x y z a : (i : ι) → α i
    hay : Not (Membership.mem (Set.univ.pi fun i => Set.Ioc (y i) (z i)) a)
    hax : LE.le x a
    haz : LE.le a z
    ⊢ Membership.mem (Set.iUnion fun i => Set.Icc x (Function.update z i (y i))) a
  -/
  simpa [not_and_or, hax, le_update_iff, haz _] using hay
  /-
    🎉 no goals
  -/


