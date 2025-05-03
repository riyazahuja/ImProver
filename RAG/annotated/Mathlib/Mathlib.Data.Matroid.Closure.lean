/-- A flat is a maximal set having a given basis  -/
@[mk_iff]
structure Flat (M : Matroid α) (F : Set α) : Prop where
  subset_of_basis_of_basis : ∀ ⦃I X⦄, M.Basis I F → M.Basis I X → X ⊆ F
  subset_ground : F ⊆ M.E


@[simp] lemma ground_flat (M : Matroid α) : M.Flat M.E :=
  ⟨fun _ _ _ ↦ Basis.subset_ground, Subset.rfl⟩


lemma Flat.iInter {ι : Type*} [Nonempty ι] {Fs : ι → Set α}
    (hFs : ∀ i, M.Flat (Fs i)) : M.Flat (⋂ i, Fs i) := by
  refine ⟨fun I X hI hIX ↦ subset_iInter fun i ↦ ?_,
    (iInter_subset _ (Classical.arbitrary _)).trans (hFs _).subset_ground⟩
  /-
    α : Type u_2
    M : Matroid α
    ι : Type u_3
    inst✝ : Nonempty ι
    Fs : ι → Set α
    hFs : ∀ (i : ι), M.Flat (Fs i)
    I X : Set α
    hI : M.Basis I (Set.iInter fun i => Fs i)
    hIX : M.Basis I X
    i : ι
    ⊢ HasSubset.Subset X (Fs i)
  -/
  obtain ⟨J, hIJ, hJ⟩ := hI.indep.subset_basis_of_subset (hI.subset.trans (iInter_subset _ i))
  /-
    case intro.intro
    α : Type u_2
    M : Matroid α
    ι : Type u_3
    inst✝ : Nonempty ι
    Fs : ι → Set α
    hFs : ∀ (i : ι), M.Flat (Fs i)
    I X : Set α
    hI : M.Basis I (Set.iInter fun i => Fs i)
    hIX : M.Basis I X
    i : ι
    J : Set α
    hIJ : M.Basis J (Fs i)
    hJ : HasSubset.Subset I J
    ⊢ HasSubset.Subset X (Fs i)
  -/
  refine subset_union_right.trans ((hFs i).1 (X := Fs i ∪ X) hIJ ?_)
  /-
    case intro.intro
    α : Type u_2
    M : Matroid α
    ι : Type u_3
    inst✝ : Nonempty ι
    Fs : ι → Set α
    hFs : ∀ (i : ι), M.Flat (Fs i)
    I X : Set α
    hI : M.Basis I (Set.iInter fun i => Fs i)
    hIX : M.Basis I X
    i : ι
    J : Set α
    hIJ : M.Basis J (Fs i)
    hJ : HasSubset.Subset I J
    ⊢ M.Basis J (Union.union (Fs i) X)
  -/
  convert hIJ.basis_union (hIX.basis_union_of_subset hIJ.indep hJ) using 1
  /-
    case h.e'_4
    α : Type u_2
    M : Matroid α
    ι : Type u_3
    inst✝ : Nonempty ι
    Fs : ι → Set α
    hFs : ∀ (i : ι), M.Flat (Fs i)
    I X : Set α
    hI : M.Basis I (Set.iInter fun i => Fs i)
    hIX : M.Basis I X
    i : ι
    J : Set α
    hIJ : M.Basis J (Fs i)
    hJ : HasSubset.Subset I J
    ⊢ Eq (Union.union (Fs i) X) (Union.union (Fs i) (Union.union J X))
  -/
  rw [← union_assoc, union_eq_self_of_subset_right hIJ.subset]
  /-
    🎉 no goals
  -/


/-- The property of being a flat gives rise to a `ClosureOperator` on the subsets of `M.E`,
in which the `IsClosed` sets correspond to `Flat`s.
(We can't define such an operator on all of `Set α`,
since this would incorrectly force `univ` to always be a flat.) -/
def subtypeClosure (M : Matroid α) : ClosureOperator (Iic M.E) :=
  ClosureOperator.ofCompletePred (fun F ↦ M.Flat F.1) fun s hs ↦ by
    /-
      ι : Type u_1
      α : Type u_2
      M✝ : Matroid α
      F X Y : Set α
      e f : α
      M : Matroid α
      s : Set ↑(Set.Iic M.E)
      hs : ∀ (a : ↑(Set.Iic M.E)), Membership.mem s a → (fun F => M.Flat ↑F) a
      ⊢ (fun F => M.Flat ↑F) (InfSet.sInf s)
    -/
    obtain (rfl | hne) := s.eq_empty_or_nonempty
      /-
        case inl
        ι : Type u_1
        α : Type u_2
        M✝ : Matroid α
        F X Y : Set α
        e f : α
        M : Matroid α
        hs : ∀ (a : ↑(Set.Iic M.E)), Membership.mem EmptyCollection.emptyCollection a  …
        ⊢ M.Flat ↑(InfSet.sInf EmptyCollection.emptyCollection)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_1
      α : Type u_2
      M✝ : Matroid α
      F X Y : Set α
      e f : α
      M : Matroid α
      s : Set ↑(Set.Iic M.E)
      hs : ∀ (a : ↑(Set.Iic M.E)), Membership.mem s a → (fun F => M.Flat ↑F) a
      hne : s.Nonempty
      ⊢ M.Flat ↑(InfSet.sInf s)
    -/
    have _ := hne.coe_sort
    /-
      case inr
      ι : Type u_1
      α : Type u_2
      M✝ : Matroid α
      F X Y : Set α
      e f : α
      M : Matroid α
      s : Set ↑(Set.Iic M.E)
      hs : ∀ (a : ↑(Set.Iic M.E)), Membership.mem s a → (fun F => M.Flat ↑F) a
      hne : s.Nonempty
      x✝ : Nonempty ↑s
      ⊢ M.Flat ↑(InfSet.sInf s)
    -/
    convert Flat.iInter (M := M) (Fs := fun (F : s) ↦ F.1.1) (fun F ↦ hs F.1 F.2)
    /-
      case h.e'_3
      ι : Type u_1
      α : Type u_2
      M✝ : Matroid α
      F X Y : Set α
      e f : α
      M : Matroid α
      s : Set ↑(Set.Iic M.E)
      hs : ∀ (a : ↑(Set.Iic M.E)), Membership.mem s a → (fun F => M.Flat ↑F) a
      hne : s.Nonempty
      x✝ : Nonempty ↑s
      ⊢ Eq (↑(InfSet.sInf s)) (Set.iInter fun i => ↑↑i)
    -/
    ext
    /-
      case h.e'_3.h
      ι : Type u_1
      α : Type u_2
      M✝ : Matroid α
      F X Y : Set α
      e f : α
      M : Matroid α
      s : Set ↑(Set.Iic M.E)
      hs : ∀ (a : ↑(Set.Iic M.E)), Membership.mem s a → (fun F => M.Flat ↑F) a
      hne : s.Nonempty
      x✝¹ : Nonempty ↑s
      x✝ : α
      ⊢ Iff (Membership.mem (↑(InfSet.sInf s)) x✝) (Membership.mem (Set.iInter fun i …
    -/
    aesop
    /-
      🎉 no goals
    -/


lemma flat_iff_isClosed : M.Flat F ↔ ∃ h : F ⊆ M.E, M.subtypeClosure.IsClosed ⟨F, h⟩ := by
  /-
    α : Type u_2
    M : Matroid α
    F : Set α
    ⊢ Iff (M.Flat F) (Exists fun h => M.subtypeClosure.IsClosed ⟨F, h⟩)
  -/
  simpa [subtypeClosure] using Flat.subset_ground
  /-
    🎉 no goals
  -/


lemma isClosed_iff_flat {F : Iic M.E} : M.subtypeClosure.IsClosed F ↔ M.Flat F := by
  /-
    α : Type u_2
    M : Matroid α
    F : ↑(Set.Iic M.E)
    ⊢ Iff (M.subtypeClosure.IsClosed F) (M.Flat ↑F)
  -/
  simp [subtypeClosure]
  /-
    🎉 no goals
  -/


/-- The closure of `X ⊆ M.E` is the intersection of all the flats of `M` containing `X`.
A set `X` that doesn't satisfy `X ⊆ M.E` has the junk value `M.closure X := M.closure (X ∩ M.E)`. -/
def closure (M : Matroid α) (X : Set α) : Set α := ⋂₀ {F | M.Flat F ∧ X ∩ M.E ⊆ F}


lemma closure_def (M : Matroid α) (X : Set α) : M.closure X = ⋂₀ {F | M.Flat F ∧ X ∩ M.E ⊆ F} := rfl


lemma closure_def' (M : Matroid α) (X : Set α) (hX : X ⊆ M.E := by aesop_mat) :
    M.closure X = ⋂₀ {F | M.Flat F ∧ X ⊆ F} := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Eq (M.closure X) (setOf fun F => And (M.Flat F) (HasSubset.Subset X F)).sInter
  -/
  rw [closure, inter_eq_self_of_subset_left hX]
  /-
    🎉 no goals
  -/


lemma closure_eq_subtypeClosure (M : Matroid α) (X : Set α) :
    M.closure X = M.subtypeClosure ⟨X ∩ M.E, inter_subset_right⟩  := by
  suffices ∀ (x : α), (∀ (t : Set α), M.Flat t → X ∩ M.E ⊆ t → x ∈ t) ↔
    (x ∈ M.E ∧ ∀ a ⊆ M.E, X ∩ M.E ⊆ a → M.Flat a → x ∈ a) by
    simpa [closure, subtypeClosure, Set.ext_iff]
  exact fun x ↦ ⟨fun h ↦ ⟨h _ M.ground_flat inter_subset_right, fun F _ hXF hF ↦ h F hF hXF⟩,
    fun ⟨_, h⟩ F hF hXF ↦ h F hF.subset_ground hXF hF⟩


@[aesop unsafe 10% (rule_sets := [Matroid])]
lemma closure_subset_ground (M : Matroid α) (X : Set α) : M.closure X ⊆ M.E :=
  sInter_subset_of_mem ⟨M.ground_flat, inter_subset_right⟩


@[simp] lemma ground_subset_closure_iff : M.E ⊆ M.closure X ↔ M.closure X = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    ⊢ Iff (HasSubset.Subset M.E (M.closure X)) (Eq (M.closure X) M.E)
  -/
  simp [M.closure_subset_ground X, subset_antisymm_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma closure_inter_ground (M : Matroid α) (X : Set α) :
    M.closure (X ∩ M.E) = M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    ⊢ Eq (M.closure (Inter.inter X M.E)) (M.closure X)
  -/
  simp_rw [closure_def, inter_assoc, inter_self]
  /-
    🎉 no goals
  -/


lemma inter_ground_subset_closure (M : Matroid α) (X : Set α) : X ∩ M.E ⊆ M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    ⊢ HasSubset.Subset (Inter.inter X M.E) (M.closure X)
  -/
  simp_rw [closure_def, subset_sInter_iff]; aesop
                                            /-
                                              🎉 no goals
                                            -/


lemma mem_closure_iff_forall_mem_flat (X : Set α) (hX : X ⊆ M.E := by aesop_mat) :
    e ∈ M.closure X ↔ ∀ F, M.Flat F → X ⊆ F → e ∈ F := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (Membership.mem (M.closure X) e) (∀ (F : Set α), M.Flat F → HasSubset.Su …
  -/
  simp_rw [M.closure_def' X, mem_sInter, mem_setOf, and_imp]
  /-
    🎉 no goals
  -/


lemma subset_closure_iff_forall_subset_flat (X : Set α) (hX : X ⊆ M.E := by aesop_mat) :
    Y ⊆ M.closure X ↔ ∀ F, M.Flat F → X ⊆ F → Y ⊆ F := by
  /-
    α : Type u_2
    M : Matroid α
    Y X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (HasSubset.Subset Y (M.closure X)) (∀ (F : Set α), M.Flat F → HasSubset. …
  -/
  simp_rw [M.closure_def' X, subset_sInter_iff, mem_setOf, and_imp]
  /-
    🎉 no goals
  -/


lemma subset_closure (M : Matroid α) (X : Set α) (hX : X ⊆ M.E := by aesop_mat) :
    X ⊆ M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ HasSubset.Subset X (M.closure X)
  -/
  simp [M.closure_def' X, subset_sInter_iff]
  /-
    🎉 no goals
  -/


lemma Flat.closure (hF : M.Flat F) : M.closure F = F :=
                            /-
                              α : Type u_2
                              M : Matroid α
                              F : Set α
                              hF : M.Flat F
                              ⊢ Membership.mem (setOf fun F_1 => And (M.Flat F_1) (HasSubset.Subset (Inter.i …
                            -/
                            /-
                              🎉 no goals
                            -/
  (sInter_subset_of_mem (by simpa)).antisymm (M.subset_closure F)
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma closure_ground (M : Matroid α) : M.closure M.E = M.E :=
                                          /-
                                            α : Type u_2
                                            M : Matroid α
                                            ⊢ HasSubset.Subset M.E M.E
                                          -/
  (M.closure_subset_ground M.E).antisymm (M.subset_closure M.E)
                                          /-
                                            🎉 no goals
                                          -/


@[simp] lemma closure_univ (M : Matroid α) : M.closure univ = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    ⊢ Eq (M.closure Set.univ) M.E
  -/
  rw [← closure_inter_ground, univ_inter, closure_ground]
  /-
    🎉 no goals
  -/


@[gcongr]
lemma closure_subset_closure (M : Matroid α) (h : X ⊆ Y) : M.closure X ⊆ M.closure Y :=
  subset_sInter (fun _ h' ↦ sInter_subset_of_mem
    ⟨h'.1, subset_trans (inter_subset_inter_left _ h) h'.2⟩)


lemma closure_mono (M : Matroid α) : Monotone M.closure :=
  fun _ _ ↦ M.closure_subset_closure


@[simp] lemma closure_closure (M : Matroid α) (X : Set α) : M.closure (M.closure X) = M.closure X :=
   /-
     α : Type u_2
     M : Matroid α
     X : Set α
     ⊢ HasSubset.Subset (M.closure X) M.E
   -/
  (M.subset_closure _).antisymm' (subset_sInter
   /-
     🎉 no goals
   -/
    (fun F hF ↦ (closure_subset_closure _ (sInter_subset_of_mem hF)).trans hF.1.closure.subset))


lemma closure_subset_closure_of_subset_closure (hXY : X ⊆ M.closure Y) :
    M.closure X ⊆ M.closure Y :=
  (M.closure_subset_closure hXY).trans_eq (M.closure_closure Y)


lemma closure_subset_closure_iff_subset_closure (hX : X ⊆ M.E := by aesop_mat) :
    M.closure X ⊆ M.closure Y ↔ X ⊆ M.closure Y :=
    /-
      α : Type u_2
      M : Matroid α
      X Y : Set α
      hX : autoParam (HasSubset.Subset X M.E) _auto✝
      ⊢ HasSubset.Subset X M.E
    -/
  ⟨(M.subset_closure X).trans, closure_subset_closure_of_subset_closure⟩
    /-
      🎉 no goals
    -/


lemma subset_closure_of_subset (M : Matroid α) (hXY : X ⊆ Y) (hY : Y ⊆ M.E := by aesop_mat) :
    X ⊆ M.closure Y :=
             /-
               α : Type u_2
               X Y : Set α
               M : Matroid α
               hXY : HasSubset.Subset X Y
               hY : autoParam (HasSubset.Subset Y M.E) _auto✝
               ⊢ HasSubset.Subset Y M.E
             -/
  hXY.trans (M.subset_closure Y)
             /-
               🎉 no goals
             -/


lemma subset_closure_of_subset' (M : Matroid α) (hXY : X ⊆ Y) (hX : X ⊆ M.E := by aesop_mat) :
    X ⊆ M.closure Y := by
  /-
    α : Type u_2
    X Y : Set α
    M : Matroid α
    hXY : HasSubset.Subset X Y
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ HasSubset.Subset X (M.closure Y)
  -/
  rw [← closure_inter_ground]; exact M.subset_closure_of_subset (subset_inter hXY hX)
                               /-
                                 🎉 no goals
                               -/


lemma exists_of_closure_ssubset (hXY : M.closure X ⊂ M.closure Y) : ∃ e ∈ Y, e ∉ M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X Y : Set α
    hXY : HasSSubset.SSubset (M.closure X) (M.closure Y)
    ⊢ Exists fun e => And (Membership.mem Y e) (Not (Membership.mem (M.closure X)  …
  -/
  by_contra! hcon
  /-
    α : Type u_2
    M : Matroid α
    X Y : Set α
    hXY : HasSSubset.SSubset (M.closure X) (M.closure Y)
    hcon : ∀ (e : α), Membership.mem Y e → Membership.mem (M.closure X) e
    ⊢ False
  -/
  exact hXY.not_subset (M.closure_subset_closure_of_subset_closure hcon)
  /-
    🎉 no goals
  -/


lemma mem_closure_of_mem (M : Matroid α) (h : e ∈ X) (hX : X ⊆ M.E := by aesop_mat) :
    e ∈ M.closure X :=
   /-
     α : Type u_2
     X : Set α
     e : α
     M : Matroid α
     h : Membership.mem X e
     hX : autoParam (HasSubset.Subset X M.E) _auto✝
     ⊢ HasSubset.Subset X M.E
   -/
  (M.subset_closure X) h
   /-
     🎉 no goals
   -/


lemma mem_closure_of_mem' (M : Matroid α) (heX : e ∈ X) (h : e ∈ M.E := by aesop_mat) :
    e ∈ M.closure X := by
  /-
    α : Type u_2
    X : Set α
    e : α
    M : Matroid α
    heX : Membership.mem X e
    h : autoParam (Membership.mem M.E e) _auto✝
    ⊢ Membership.mem (M.closure X) e
  -/
  rw [← closure_inter_ground]
  /-
    α : Type u_2
    X : Set α
    e : α
    M : Matroid α
    heX : Membership.mem X e
    h : autoParam (Membership.mem M.E e) _auto✝
    ⊢ Membership.mem (M.closure (Inter.inter X M.E)) e
  -/
  exact M.mem_closure_of_mem ⟨heX, h⟩
  /-
    🎉 no goals
  -/


lemma not_mem_of_mem_diff_closure (he : e ∈ M.E \ M.closure X) : e ∉ X :=
  fun heX ↦ he.2 <| M.mem_closure_of_mem' heX he.1


@[aesop unsafe 10% (rule_sets := [Matroid])]
lemma mem_ground_of_mem_closure (he : e ∈ M.closure X) : e ∈ M.E :=
  (M.closure_subset_ground _) he


lemma closure_iUnion_closure_eq_closure_iUnion (M : Matroid α) (Xs : ι → Set α) :
    M.closure (⋃ i, M.closure (Xs i)) = M.closure (⋃ i, Xs i) := by
  /-
    ι : Type u_1
    α : Type u_2
    M : Matroid α
    Xs : ι → Set α
    ⊢ Eq (M.closure (Set.iUnion fun i => M.closure (Xs i))) (M.closure (Set.iUnion …
  -/
  simp_rw [closure_eq_subtypeClosure, iUnion_inter, Subtype.coe_inj]
  /-
    ι : Type u_1
    α : Type u_2
    M : Matroid α
    Xs : ι → Set α
    ⊢ Eq (M.subtypeClosure ⟨Set.iUnion fun i => Inter.inter (↑(M.subtypeClosure ⟨I …
  -/
  convert M.subtypeClosure.closure_iSup_closure (fun i ↦ ⟨Xs i ∩ M.E, inter_subset_right⟩) <;>
  /-
    case h.e'_2.h.e'_6.h.e'_3
    ι : Type u_1
    α : Type u_2
    M : Matroid α
    Xs : ι → Set α
    ⊢ Eq (Set.iUnion fun i => Inter.inter (↑(M.subtypeClosure ⟨Inter.inter (Xs i)  …
  -/
  /-
    🎉 no goals
  -/
  simp [← iUnion_inter, subtypeClosure]
  /-
    🎉 no goals
  -/


lemma closure_iUnion_congr (Xs Ys : ι → Set α) (h : ∀ i, M.closure (Xs i) = M.closure (Ys i)) :
    M.closure (⋃ i, Xs i) = M.closure (⋃ i, Ys i) := by
  /-
    ι : Type u_1
    α : Type u_2
    M : Matroid α
    Xs Ys : ι → Set α
    h : ∀ (i : ι), Eq (M.closure (Xs i)) (M.closure (Ys i))
    ⊢ Eq (M.closure (Set.iUnion fun i => Xs i)) (M.closure (Set.iUnion fun i => Ys …
  -/
  simp [h, ← M.closure_iUnion_closure_eq_closure_iUnion]
  /-
    🎉 no goals
  -/


lemma closure_biUnion_closure_eq_closure_sUnion (M : Matroid α) (Xs : Set (Set α)) :
    M.closure (⋃ X ∈ Xs, M.closure X) = M.closure (⋃₀ Xs) := by
  /-
    α : Type u_2
    M : Matroid α
    Xs : Set (Set α)
    ⊢ Eq (M.closure (Set.iUnion fun X => Set.iUnion fun h => M.closure X)) (M.clos …
  -/
  rw [sUnion_eq_iUnion, biUnion_eq_iUnion, closure_iUnion_closure_eq_closure_iUnion]
  /-
    🎉 no goals
  -/


lemma closure_biUnion_closure_eq_closure_biUnion (M : Matroid α) (Xs : ι → Set α) (A : Set ι) :
    M.closure (⋃ i ∈ A, M.closure (Xs i)) = M.closure (⋃ i ∈ A, Xs i) := by
  /-
    ι : Type u_1
    α : Type u_2
    M : Matroid α
    Xs : ι → Set α
    A : Set ι
    ⊢ Eq (M.closure (Set.iUnion fun i => Set.iUnion fun h => M.closure (Xs i))) (M …
  -/
  rw [biUnion_eq_iUnion, M.closure_iUnion_closure_eq_closure_iUnion, biUnion_eq_iUnion]
  /-
    🎉 no goals
  -/


lemma closure_biUnion_congr (M : Matroid α) (Xs Ys : ι → Set α) (A : Set ι)
    (h : ∀ i ∈ A, M.closure (Xs i) = M.closure (Ys i)) :
    M.closure (⋃ i ∈ A, Xs i) = M.closure (⋃ i ∈ A, Ys i) := by
  rw [← closure_biUnion_closure_eq_closure_biUnion, iUnion₂_congr h,
    closure_biUnion_closure_eq_closure_biUnion]


lemma closure_closure_union_closure_eq_closure_union (M : Matroid α) (X Y : Set α) :
    M.closure (M.closure X ∪ M.closure Y) = M.closure (X ∪ Y) := by
  /-
    α : Type u_2
    M : Matroid α
    X Y : Set α
    ⊢ Eq (M.closure (Union.union (M.closure X) (M.closure Y))) (M.closure (Union.u …
  -/
  rw [eq_comm, union_eq_iUnion, ← closure_iUnion_closure_eq_closure_iUnion, union_eq_iUnion]
  /-
    α : Type u_2
    M : Matroid α
    X Y : Set α
    ⊢ Eq (M.closure (Set.iUnion fun i => M.closure (cond i X Y))) (M.closure (Set. …
  -/
  simp_rw [Bool.cond_eq_ite, apply_ite]
  /-
    🎉 no goals
  -/


@[simp] lemma closure_union_closure_right_eq (M : Matroid α) (X Y : Set α) :
    M.closure (X ∪ M.closure Y) = M.closure (X ∪ Y) := by
  rw [← closure_closure_union_closure_eq_closure_union, closure_closure,
    closure_closure_union_closure_eq_closure_union]


@[simp] lemma closure_union_closure_left_eq (M : Matroid α) (X Y : Set α) :
    M.closure (M.closure X ∪ Y) = M.closure (X ∪ Y) := by
  rw [← closure_closure_union_closure_eq_closure_union, closure_closure,
    closure_closure_union_closure_eq_closure_union]


@[simp] lemma closure_insert_closure_eq_closure_insert (M : Matroid α) (e : α) (X : Set α) :
    M.closure (insert e (M.closure X)) = M.closure (insert e X) := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    X : Set α
    ⊢ Eq (M.closure (Insert.insert e (M.closure X))) (M.closure (Insert.insert e X))
  -/
  simp_rw [← singleton_union, closure_union_closure_right_eq]
  /-
    🎉 no goals
  -/


@[simp] lemma closure_union_closure_empty_eq (M : Matroid α) (X : Set α) :
    M.closure X ∪ M.closure ∅ = M.closure X :=
  union_eq_self_of_subset_right (M.closure_subset_closure (empty_subset _))


@[simp] lemma closure_empty_union_closure_eq (M : Matroid α) (X : Set α) :
    M.closure ∅ ∪ M.closure X = M.closure X :=
  union_eq_self_of_subset_left (M.closure_subset_closure (empty_subset _))


lemma closure_insert_eq_of_mem_closure (he : e ∈ M.closure X) :
    M.closure (insert e X) = M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e : α
    he : Membership.mem (M.closure X) e
    ⊢ Eq (M.closure (Insert.insert e X)) (M.closure X)
  -/
  rw [← closure_insert_closure_eq_closure_insert, insert_eq_of_mem he, closure_closure]
  /-
    🎉 no goals
  -/


lemma mem_closure_self (M : Matroid α) (e : α) (he : e ∈ M.E := by aesop_mat) : e ∈ M.closure {e} :=
  /-
    α : Type u_2
    M : Matroid α
    e : α
    he : autoParam (Membership.mem M.E e) _auto✝
    ⊢ Membership.mem M.E e
  -/
  mem_closure_of_mem' M rfl
  /-
    🎉 no goals
  -/


lemma Indep.closure_eq_setOf_basis_insert (hI : M.Indep I) :
    M.closure I = {x | M.Basis I (insert x I)} := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    ⊢ Eq (M.closure I) (setOf fun x => M.Basis I (Insert.insert x I))
  -/
  set F := {x | M.Basis I (insert x I)}
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
    ⊢ Eq (M.closure I) F
  -/
  have hIF : M.Basis I F := hI.basis_setOf_insert_basis

  have hF : M.Flat F := by
    refine ⟨fun J X hJF hJX e heX ↦ show M.Basis _ _ from ?_, hIF.subset_ground⟩
    exact (hIF.basis_of_basis_of_subset_of_subset (hJX.basis_union hJF) hJF.subset
      (hIF.subset.trans subset_union_right)).basis_subset (subset_insert _ _)
      (insert_subset (Or.inl heX) (hIF.subset.trans subset_union_right))

  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
    hIF : M.Basis I F
    hF : M.Flat F
    ⊢ Eq (M.closure I) F
  -/
  rw [subset_antisymm_iff, closure_def, subset_sInter_iff, and_iff_right (sInter_subset_of_mem _)]
    /-
      α : Type u_2
      M : Matroid α
      I : Set α
      hI : M.Indep I
      F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
      hIF : M.Basis I F
      hF : M.Flat F
      ⊢ ∀ (t' : Set α), Membership.mem (setOf fun F => And (M.Flat F) (HasSubset.Sub …
    -/
  · rintro F' ⟨hF', hIF'⟩ e (he : M.Basis I (insert e I))
    /-
      case intro
      α : Type u_2
      M : Matroid α
      I : Set α
      hI : M.Indep I
      F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
      hIF : M.Basis I F
      hF : M.Flat F
      F' : Set α
      hF' : M.Flat F'
      hIF' : HasSubset.Subset (Inter.inter I M.E) F'
      e : α
      he : M.Basis I (Insert.insert e I)
      ⊢ Membership.mem F' e
    -/
    rw [inter_eq_left.mpr (hIF.subset.trans hIF.subset_ground)] at hIF'
    /-
      case intro
      α : Type u_2
      M : Matroid α
      I : Set α
      hI : M.Indep I
      F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
      hIF : M.Basis I F
      hF : M.Flat F
      F' : Set α
      hF' : M.Flat F'
      hIF' : HasSubset.Subset I F'
      e : α
      he : M.Basis I (Insert.insert e I)
      ⊢ Membership.mem F' e
    -/
    obtain ⟨J, hJ, hIJ⟩ := hI.subset_basis_of_subset hIF' hF'.2
    /-
      case intro.intro.intro
      α : Type u_2
      M : Matroid α
      I : Set α
      hI : M.Indep I
      F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
      hIF : M.Basis I F
      hF : M.Flat F
      F' : Set α
      hF' : M.Flat F'
      hIF' : HasSubset.Subset I F'
      e : α
      he : M.Basis I (Insert.insert e I)
      J : Set α
      hJ : M.Basis J F'
      hIJ : HasSubset.Subset I J
      ⊢ Membership.mem F' e
    -/
    exact (hF'.1 hJ (he.basis_union_of_subset hJ.indep hIJ)) (Or.inr (mem_insert _ _))
    /-
      🎉 no goals
    -/
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    F : Set α := setOf fun x => M.Basis I (Insert.insert x I)
    hIF : M.Basis I F
    hF : M.Flat F
    ⊢ Membership.mem (setOf fun F => And (M.Flat F) (HasSubset.Subset (Inter.inter …
  -/
  exact ⟨hF, inter_subset_left.trans hIF.subset⟩
  /-
    🎉 no goals
  -/


lemma Indep.insert_basis_iff_mem_closure (hI : M.Indep I) :
    M.Basis I (insert e I) ↔ e ∈ M.closure I := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    ⊢ Iff (M.Basis I (Insert.insert e I)) (Membership.mem (M.closure I) e)
  -/
  rw [hI.closure_eq_setOf_basis_insert, mem_setOf]
  /-
    🎉 no goals
  -/


lemma Indep.basis_closure (hI : M.Indep I) : M.Basis I (M.closure I) := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    ⊢ M.Basis I (M.closure I)
  -/
  rw [hI.closure_eq_setOf_basis_insert]; exact hI.basis_setOf_insert_basis
                                         /-
                                           🎉 no goals
                                         -/


lemma Basis.closure_eq_closure (h : M.Basis I X) : M.closure I = M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    h : M.Basis I X
    ⊢ Eq (M.closure I) (M.closure X)
  -/
  refine subset_antisymm (M.closure_subset_closure h.subset) ?_
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    h : M.Basis I X
    ⊢ HasSubset.Subset (M.closure X) (M.closure I)
  -/
  rw [← M.closure_closure I, h.indep.closure_eq_setOf_basis_insert]
  exact M.closure_subset_closure fun e he ↦ (h.basis_subset (subset_insert _ _)
    (insert_subset he h.subset))


lemma Basis.closure_eq_right (h : M.Basis I (M.closure X)) : M.closure I = M.closure X :=
  M.closure_closure X ▸ h.closure_eq_closure


lemma Basis'.closure_eq_closure (h : M.Basis' I X) : M.closure I = M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    h : M.Basis' I X
    ⊢ Eq (M.closure I) (M.closure X)
  -/
  rw [← closure_inter_ground _ X, h.basis_inter_ground.closure_eq_closure]
  /-
    🎉 no goals
  -/


lemma Basis.subset_closure (h : M.Basis I X) : X ⊆ M.closure I := by
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    h : M.Basis I X
    ⊢ HasSubset.Subset X (M.closure I)
  -/
  rw [← closure_subset_closure_iff_subset_closure, h.closure_eq_closure]
  /-
    🎉 no goals
  -/


lemma Basis'.basis_closure_right (h : M.Basis' I X) : M.Basis I (M.closure X) := by
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    h : M.Basis' I X
    ⊢ M.Basis I (M.closure X)
  -/
  rw [← h.closure_eq_closure]; exact h.indep.basis_closure
                               /-
                                 🎉 no goals
                               -/


lemma Basis.basis_closure_right (h : M.Basis I X) : M.Basis I (M.closure X) :=
  h.basis'.basis_closure_right


lemma Indep.mem_closure_iff (hI : M.Indep I) :
    x ∈ M.closure I ↔ M.Dep (insert x I) ∨ x ∈ I := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    x : α
    hI : M.Indep I
    ⊢ Iff (Membership.mem (M.closure I) x) (Or (M.Dep (Insert.insert x I)) (Member …
  -/
  rwa [hI.closure_eq_setOf_basis_insert, mem_setOf, basis_insert_iff]
  /-
    🎉 no goals
  -/


lemma Indep.mem_closure_iff' (hI : M.Indep I) :
    x ∈ M.closure I ↔ x ∈ M.E ∧ (M.Indep (insert x I) → x ∈ I) := by
  rw [hI.mem_closure_iff, dep_iff, insert_subset_iff, and_iff_left hI.subset_ground,
    imp_iff_not_or]
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    x : α
    hI : M.Indep I
    ⊢ Iff (Or (And (Not (M.Indep (Insert.insert x I))) (Membership.mem M.E x)) (Me …
  -/
  have := hI.subset_ground
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    x : α
    hI : M.Indep I
    this : HasSubset.Subset I M.E
    ⊢ Iff (Or (And (Not (M.Indep (Insert.insert x I))) (Membership.mem M.E x)) (Me …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma Indep.insert_dep_iff (hI : M.Indep I) : M.Dep (insert e I) ↔ e ∈ M.closure I \ I := by
  rw [mem_diff, hI.mem_closure_iff, or_and_right, and_not_self_iff, or_false,
    iff_self_and, imp_not_comm]
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    ⊢ Membership.mem I e → Not (M.Dep (Insert.insert e I))
  -/
  intro heI; rw [insert_eq_of_mem heI]; exact hI.not_dep
                                        /-
                                          🎉 no goals
                                        -/


lemma Indep.mem_closure_iff_of_not_mem (hI : M.Indep I) (heI : e ∉ I) :
    e ∈ M.closure I ↔ M.Dep (insert e I) := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    heI : Not (Membership.mem I e)
    ⊢ Iff (Membership.mem (M.closure I) e) (M.Dep (Insert.insert e I))
  -/
  rw [hI.insert_dep_iff, mem_diff, and_iff_left heI]
  /-
    🎉 no goals
  -/


lemma Indep.not_mem_closure_iff (hI : M.Indep I) (he : e ∈ M.E := by aesop_mat) :
    e ∉ M.closure I ↔ M.Indep (insert e I) ∧ e ∉ I := by
  rw [hI.mem_closure_iff, dep_iff, insert_subset_iff, and_iff_right he,
                                    /-
                                      α : Type u_2
                                      M : Matroid α
                                      e : α
                                      I : Set α
                                      hI : M.Indep I
                                      he : autoParam (Membership.mem M.E e) _auto✝
                                      ⊢ Iff (Not (Or (Not (M.Indep (Insert.insert e I))) (Membership.mem I e))) (And …
                                    -/
    and_iff_left hI.subset_ground]; tauto
                                    /-
                                      🎉 no goals
                                    -/


lemma Indep.not_mem_closure_iff_of_not_mem (hI : M.Indep I) (heI : e ∉ I)
    (he : e ∈ M.E := by aesop_mat) : e ∉ M.closure I ↔ M.Indep (insert e I) := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    heI : Not (Membership.mem I e)
    he : autoParam (Membership.mem M.E e) _auto✝
    ⊢ Iff (Not (Membership.mem (M.closure I) e)) (M.Indep (Insert.insert e I))
  -/
  rw [hI.not_mem_closure_iff, and_iff_left heI]
  /-
    🎉 no goals
  -/


lemma Indep.insert_indep_iff_of_not_mem (hI : M.Indep I) (heI : e ∉ I) :
    M.Indep (insert e I) ↔ e ∈ M.E \ M.closure I := by
  rw [mem_diff, hI.mem_closure_iff_of_not_mem heI, dep_iff, not_and, not_imp_not, insert_subset_iff,
    and_iff_left hI.subset_ground]
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    heI : Not (Membership.mem I e)
    ⊢ Iff (M.Indep (Insert.insert e I)) (And (Membership.mem M.E e) (Membership.me …
  -/
  exact ⟨fun h ↦ ⟨h.subset_ground (mem_insert e I), fun _ ↦ h⟩, fun h ↦ h.2 h.1⟩
  /-
    🎉 no goals
  -/


lemma Indep.insert_indep_iff (hI : M.Indep I) :
    M.Indep (insert e I) ↔ e ∈ M.E \ M.closure I ∨ e ∈ I := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    ⊢ Iff (M.Indep (Insert.insert e I)) (Or (Membership.mem (SDiff.sdiff M.E (M.cl …
  -/
  obtain (h | h) := em (e ∈ I)
    /-
      case inl
      α : Type u_2
      M : Matroid α
      e : α
      I : Set α
      hI : M.Indep I
      h : Membership.mem I e
      ⊢ Iff (M.Indep (Insert.insert e I)) (Or (Membership.mem (SDiff.sdiff M.E (M.cl …
    -/
  · simp_rw [insert_eq_of_mem h, iff_true_intro hI, true_iff, iff_true_intro h, or_true]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : M.Indep I
    h : Not (Membership.mem I e)
    ⊢ Iff (M.Indep (Insert.insert e I)) (Or (Membership.mem (SDiff.sdiff M.E (M.cl …
  -/
  rw [hI.insert_indep_iff_of_not_mem h, or_iff_left h]
  /-
    🎉 no goals
  -/


lemma insert_indep_iff : M.Indep (insert e I) ↔ M.Indep I ∧ (e ∉ I → e ∈ M.E \ M.closure I) := by
  /-
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    ⊢ Iff (M.Indep (Insert.insert e I)) (And (M.Indep I) (Not (Membership.mem I e) …
  -/
  by_cases hI : M.Indep I
    /-
      case pos
      α : Type u_2
      M : Matroid α
      e : α
      I : Set α
      hI : M.Indep I
      ⊢ Iff (M.Indep (Insert.insert e I)) (And (M.Indep I) (Not (Membership.mem I e) …
    -/
  · rw [hI.insert_indep_iff, and_iff_right hI, or_iff_not_imp_right]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_2
    M : Matroid α
    e : α
    I : Set α
    hI : Not (M.Indep I)
    ⊢ Iff (M.Indep (Insert.insert e I)) (And (M.Indep I) (Not (Membership.mem I e) …
  -/
  simp [hI, show ¬ M.Indep (insert e I) from fun h ↦ hI <| h.subset <| subset_insert _ _]
  /-
    🎉 no goals
  -/


/-- This can be used for rewriting if the LHS is inside a binder and whether `f = e` is unknown.-/
lemma Indep.insert_diff_indep_iff (hI : M.Indep (I \ {e})) (heI : e ∈ I) :
    M.Indep (insert f I \ {e}) ↔ f ∈ M.E \ M.closure (I \ {e}) ∨ f ∈ I := by
  /-
    α : Type u_2
    M : Matroid α
    e f : α
    I : Set α
    hI : M.Indep (SDiff.sdiff I (Singleton.singleton e))
    heI : Membership.mem I e
    ⊢ Iff (M.Indep (SDiff.sdiff (Insert.insert f I) (Singleton.singleton e))) (Or  …
  -/
  obtain rfl | hne := eq_or_ne e f
    /-
      case inl
      α : Type u_2
      M : Matroid α
      e : α
      I : Set α
      hI : M.Indep (SDiff.sdiff I (Singleton.singleton e))
      heI : Membership.mem I e
      ⊢ Iff (M.Indep (SDiff.sdiff (Insert.insert e I) (Singleton.singleton e))) (Or  …
    -/
  · simp [hI, heI]
    /-
      🎉 no goals
    -/
  rw [← insert_diff_singleton_comm hne.symm, hI.insert_indep_iff, mem_diff_singleton,
    and_iff_left hne.symm]


lemma Indep.basis_of_subset_of_subset_closure (hI : M.Indep I) (hIX : I ⊆ X)
    (hXI : X ⊆ M.closure I) : M.Basis I X :=
  hI.basis_closure.basis_subset hIX hXI


lemma basis_iff_indep_subset_closure : M.Basis I X ↔ M.Indep I ∧ I ⊆ X ∧ X ⊆ M.closure I :=
  ⟨fun h ↦ ⟨h.indep, h.subset, h.subset_closure⟩,
    fun h ↦ h.1.basis_of_subset_of_subset_closure h.2.1 h.2.2⟩


lemma Indep.base_of_ground_subset_closure (hI : M.Indep I) (h : M.E ⊆ M.closure I) : M.Base I := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    h : HasSubset.Subset M.E (M.closure I)
    ⊢ M.Base I
  -/
  rw [← basis_ground_iff]; exact hI.basis_of_subset_of_subset_closure hI.subset_ground h
                           /-
                             🎉 no goals
                           -/


lemma Base.closure_eq (hB : M.Base B) : M.closure B = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    B : Set α
    hB : M.Base B
    ⊢ Eq (M.closure B) M.E
  -/
  rw [← basis_ground_iff] at hB; rw [hB.closure_eq_closure, closure_ground]
                                 /-
                                   🎉 no goals
                                 -/


lemma Base.closure_of_superset (hB : M.Base B) (hBX : B ⊆ X) : M.closure X = M.E :=
  (M.closure_subset_ground _).antisymm (hB.closure_eq ▸ M.closure_subset_closure hBX)


lemma base_iff_indep_closure_eq : M.Base B ↔ M.Indep B ∧ M.closure B = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    B : Set α
    ⊢ Iff (M.Base B) (And (M.Indep B) (Eq (M.closure B) M.E))
  -/
  rw [← basis_ground_iff, basis_iff_indep_subset_closure, and_congr_right_iff]
  exact fun hI ↦ ⟨fun h ↦ (M.closure_subset_ground _).antisymm h.2,
    fun h ↦ ⟨(M.subset_closure B).trans_eq h, h.symm.subset⟩⟩


lemma Indep.base_iff_ground_subset_closure (hI : M.Indep I) : M.Base I ↔ M.E ⊆ M.closure I :=
  ⟨fun h ↦ h.closure_eq.symm.subset, hI.base_of_ground_subset_closure⟩


lemma Indep.closure_inter_eq_self_of_subset (hI : M.Indep I) (hJI : J ⊆ I) :
    M.closure J ∩ I = J := by
  /-
    α : Type u_2
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJI : HasSubset.Subset J I
    ⊢ Eq (Inter.inter (M.closure J) I) J
  -/
  have hJ := hI.subset hJI
  /-
    α : Type u_2
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJI : HasSubset.Subset J I
    hJ : M.Indep J
    ⊢ Eq (Inter.inter (M.closure J) I) J
  -/
  rw [subset_antisymm_iff, and_iff_left (subset_inter (M.subset_closure _) hJI)]
  /-
    α : Type u_2
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJI : HasSubset.Subset J I
    hJ : M.Indep J
    ⊢ HasSubset.Subset (Inter.inter (M.closure J) I) J
  -/
  rintro e ⟨heJ, heI⟩
  /-
    case intro
    α : Type u_2
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJI : HasSubset.Subset J I
    hJ : M.Indep J
    e : α
    heJ : Membership.mem (M.closure J) e
    heI : Membership.mem I e
    ⊢ Membership.mem J e
  -/
  exact hJ.basis_closure.mem_of_insert_indep heJ (hI.subset (insert_subset heI hJI))
  /-
    🎉 no goals
  -/


/-- For a nonempty collection of subsets of a given independent set,
the closure of the intersection is the intersection of the closure. -/
lemma Indep.closure_sInter_eq_biInter_closure_of_forall_subset {Js : Set (Set α)} (hI : M.Indep I)
    (hne : Js.Nonempty) (hIs : ∀ J ∈ Js, J ⊆ I) : M.closure (⋂₀ Js) = (⋂ J ∈ Js, M.closure J)  := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    Js : Set (Set α)
    hI : M.Indep I
    hne : Js.Nonempty
    hIs : ∀ (J : Set α), Membership.mem Js J → HasSubset.Subset J I
    ⊢ Eq (M.closure Js.sInter) (Set.iInter fun J => Set.iInter fun h => M.closure J)
  -/
  rw [subset_antisymm_iff, subset_iInter₂_iff]
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    Js : Set (Set α)
    hI : M.Indep I
    hne : Js.Nonempty
    hIs : ∀ (J : Set α), Membership.mem Js J → HasSubset.Subset J I
    ⊢ And (∀ (i : Set α), Membership.mem Js i → HasSubset.Subset (M.closure Js.sIn …
  -/
  have hiX : ⋂₀ Js ⊆ I := (sInter_subset_of_mem hne.some_mem).trans (hIs _ hne.some_mem)
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    Js : Set (Set α)
    hI : M.Indep I
    hne : Js.Nonempty
    hIs : ∀ (J : Set α), Membership.mem Js J → HasSubset.Subset J I
    hiX : HasSubset.Subset Js.sInter I
    ⊢ And (∀ (i : Set α), Membership.mem Js i → HasSubset.Subset (M.closure Js.sIn …
  -/
  have hiI := hI.subset hiX
  refine ⟨ fun X hX ↦ M.closure_subset_closure (sInter_subset_of_mem hX),
    fun e he ↦ by_contra fun he' ↦ ?_⟩
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    Js : Set (Set α)
    hI : M.Indep I
    hne : Js.Nonempty
    hIs : ∀ (J : Set α), Membership.mem Js J → HasSubset.Subset J I
    hiX : HasSubset.Subset Js.sInter I
    hiI : M.Indep Js.sInter
    e : α
    he : Membership.mem (Set.iInter fun J => Set.iInter fun h => M.closure J) e
    he' : Not (Membership.mem (M.closure Js.sInter) e)
    ⊢ False
  -/
  rw [mem_iInter₂] at he
  have heEI : e ∈ M.E \ I := by
    refine ⟨M.closure_subset_ground _ (he _ hne.some_mem), fun heI ↦ he' ?_⟩
    refine mem_closure_of_mem _ (fun X hX' ↦ ?_) hiI.subset_ground
    rw [← hI.closure_inter_eq_self_of_subset (hIs X hX')]
    exact ⟨he X hX', heI⟩

  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    Js : Set (Set α)
    hI : M.Indep I
    hne : Js.Nonempty
    hIs : ∀ (J : Set α), Membership.mem Js J → HasSubset.Subset J I
    hiX : HasSubset.Subset Js.sInter I
    hiI : M.Indep Js.sInter
    e : α
    he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
    he' : Not (Membership.mem (M.closure Js.sInter) e)
    heEI : Membership.mem (SDiff.sdiff M.E I) e
    ⊢ False
  -/
  rw [hiI.not_mem_closure_iff_of_not_mem (not_mem_subset hiX heEI.2)] at he'
  obtain ⟨J, hJI, heJ⟩ := he'.subset_basis_of_subset (insert_subset_insert hiX)
    (insert_subset heEI.1 hI.subset_ground)

  have hIb : M.Basis I (insert e I) := by
    rw [hI.insert_basis_iff_mem_closure]
    exact (M.closure_subset_closure (hIs _ hne.some_mem)) (he _ hne.some_mem)

  /-
    case intro.intro
    α : Type u_2
    M : Matroid α
    I : Set α
    Js : Set (Set α)
    hI : M.Indep I
    hne : Js.Nonempty
    hIs : ∀ (J : Set α), Membership.mem Js J → HasSubset.Subset J I
    hiX : HasSubset.Subset Js.sInter I
    hiI : M.Indep Js.sInter
    e : α
    he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
    he' : M.Indep (Insert.insert e Js.sInter)
    heEI : Membership.mem (SDiff.sdiff M.E I) e
    J : Set α
    hJI : M.Basis J (Insert.insert e I)
    heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
    hIb : M.Basis I (Insert.insert e I)
    ⊢ False
  -/
  obtain ⟨f, hfIJ, hfb⟩ :=  hJI.exchange hIb ⟨heJ (mem_insert e _), heEI.2⟩
  obtain rfl := hI.eq_of_basis (hfb.basis_subset (insert_subset hfIJ.1
    (by (rw [diff_subset_iff, singleton_union]; exact hJI.subset))) (subset_insert _ _))

  /-
    case intro.intro.intro.intro
    α : Type u_2
    M : Matroid α
    Js : Set (Set α)
    hne : Js.Nonempty
    hiI : M.Indep Js.sInter
    e : α
    he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
    he' : M.Indep (Insert.insert e Js.sInter)
    J : Set α
    heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
    f : α
    hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
    hIs : ∀ (J_1 : Set α), Membership.mem Js J_1 → HasSubset.Subset J_1 (Insert.in …
    hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
    heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
    hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
    hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
    hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
    hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
    ⊢ False
  -/
  refine hfIJ.2 (heJ (mem_insert_of_mem _ fun X hX' ↦ by_contra fun hfX ↦ ?_))

  /-
    case intro.intro.intro.intro
    α : Type u_2
    M : Matroid α
    Js : Set (Set α)
    hne : Js.Nonempty
    hiI : M.Indep Js.sInter
    e : α
    he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
    he' : M.Indep (Insert.insert e Js.sInter)
    J : Set α
    heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
    f : α
    hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
    hIs : ∀ (J_1 : Set α), Membership.mem Js J_1 → HasSubset.Subset J_1 (Insert.in …
    hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
    heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
    hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
    hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
    hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
    hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
    X : Set α
    hX' : Membership.mem Js X
    hfX : Not (Membership.mem X f)
    ⊢ False
  -/
  obtain (hd | heX) := ((hI.subset (hIs X hX')).mem_closure_iff).mp (he _ hX')
    /-
      case intro.intro.intro.intro.inl
      α : Type u_2
      M : Matroid α
      Js : Set (Set α)
      hne : Js.Nonempty
      hiI : M.Indep Js.sInter
      e : α
      he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
      he' : M.Indep (Insert.insert e Js.sInter)
      J : Set α
      heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
      f : α
      hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
      hIs : ∀ (J_1 : Set α), Membership.mem Js J_1 → HasSubset.Subset J_1 (Insert.in …
      hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
      heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
      hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
      hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
      hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      X : Set α
      hX' : Membership.mem Js X
      hfX : Not (Membership.mem X f)
      hd : M.Dep (Insert.insert e X)
      ⊢ False
    -/
  · refine (hJI.indep.subset (insert_subset (heJ (mem_insert _ _)) ?_)).not_dep hd
    /-
      case intro.intro.intro.intro.inl
      α : Type u_2
      M : Matroid α
      Js : Set (Set α)
      hne : Js.Nonempty
      hiI : M.Indep Js.sInter
      e : α
      he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
      he' : M.Indep (Insert.insert e Js.sInter)
      J : Set α
      heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
      f : α
      hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
      hIs : ∀ (J_1 : Set α), Membership.mem Js J_1 → HasSubset.Subset J_1 (Insert.in …
      hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
      heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
      hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
      hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
      hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      X : Set α
      hX' : Membership.mem Js X
      hfX : Not (Membership.mem X f)
      hd : M.Dep (Insert.insert e X)
      ⊢ HasSubset.Subset X J
    -/
    specialize hIs _ hX'
    /-
      case intro.intro.intro.intro.inl
      α : Type u_2
      M : Matroid α
      Js : Set (Set α)
      hne : Js.Nonempty
      hiI : M.Indep Js.sInter
      e : α
      he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
      he' : M.Indep (Insert.insert e Js.sInter)
      J : Set α
      heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
      f : α
      hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
      hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
      heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
      hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
      hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
      hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      X : Set α
      hX' : Membership.mem Js X
      hfX : Not (Membership.mem X f)
      hd : M.Dep (Insert.insert e X)
      hIs : HasSubset.Subset X (Insert.insert f (SDiff.sdiff J (Singleton.singleton  …
      ⊢ HasSubset.Subset X J
    -/
    rw [← singleton_union, ← diff_subset_iff, diff_singleton_eq_self hfX] at hIs
    /-
      case intro.intro.intro.intro.inl
      α : Type u_2
      M : Matroid α
      Js : Set (Set α)
      hne : Js.Nonempty
      hiI : M.Indep Js.sInter
      e : α
      he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
      he' : M.Indep (Insert.insert e Js.sInter)
      J : Set α
      heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
      f : α
      hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
      hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
      heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
      hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
      hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
      hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
      X : Set α
      hX' : Membership.mem Js X
      hfX : Not (Membership.mem X f)
      hd : M.Dep (Insert.insert e X)
      hIs : HasSubset.Subset X (SDiff.sdiff J (Singleton.singleton e))
      ⊢ HasSubset.Subset X J
    -/
    exact hIs.trans diff_subset
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.inr
    α : Type u_2
    M : Matroid α
    Js : Set (Set α)
    hne : Js.Nonempty
    hiI : M.Indep Js.sInter
    e : α
    he : ∀ (i : Set α), Membership.mem Js i → Membership.mem (M.closure i) e
    he' : M.Indep (Insert.insert e Js.sInter)
    J : Set α
    heJ : HasSubset.Subset (Insert.insert e Js.sInter) J
    f : α
    hI : M.Indep (Insert.insert f (SDiff.sdiff J (Singleton.singleton e)))
    hIs : ∀ (J_1 : Set α), Membership.mem Js J_1 → HasSubset.Subset J_1 (Insert.in …
    hiX : HasSubset.Subset Js.sInter (Insert.insert f (SDiff.sdiff J (Singleton.si …
    heEI : Membership.mem (SDiff.sdiff M.E (Insert.insert f (SDiff.sdiff J (Single …
    hJI : M.Basis J (Insert.insert e (Insert.insert f (SDiff.sdiff J (Singleton.si …
    hIb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
    hfIJ : Membership.mem (SDiff.sdiff (Insert.insert f (SDiff.sdiff J (Singleton. …
    hfb : M.Basis (Insert.insert f (SDiff.sdiff J (Singleton.singleton e))) (Inser …
    X : Set α
    hX' : Membership.mem Js X
    hfX : Not (Membership.mem X f)
    heX : Membership.mem X e
    ⊢ False
  -/
  exact heEI.2 (hIs _ hX' heX)
  /-
    🎉 no goals
  -/


lemma closure_iInter_eq_iInter_closure_of_iUnion_indep [hι : Nonempty ι] (Is : ι → Set α)
    (h : M.Indep (⋃ i, Is i)) : M.closure (⋂ i, Is i) = (⋂ i, M.closure (Is i)) := by
  convert h.closure_sInter_eq_biInter_closure_of_forall_subset (range_nonempty Is)
    (by simp [subset_iUnion])
  /-
    case h.e'_3
    α : Type u_2
    M : Matroid α
    ι : Sort u_3
    hι : Nonempty ι
    Is : ι → Set α
    h : M.Indep (Set.iUnion fun i => Is i)
    ⊢ Eq (Set.iInter fun i => M.closure (Is i)) (Set.iInter fun J => Set.iInter fu …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma closure_sInter_eq_biInter_closure_of_sUnion_indep (Is : Set (Set α)) (hIs : Is.Nonempty)
    (h : M.Indep (⋃₀ Is)) :  M.closure (⋂₀ Is) = (⋂ I ∈ Is, M.closure I) :=
  h.closure_sInter_eq_biInter_closure_of_forall_subset hIs (fun _ ↦ subset_sUnion_of_mem)


lemma closure_biInter_eq_biInter_closure_of_biUnion_indep {ι : Type*} {A : Set ι} (hA : A.Nonempty)
    {I : ι → Set α} (h : M.Indep (⋃ i ∈ A, I i)) :
    M.closure (⋂ i ∈ A, I i) = ⋂ i ∈ A, M.closure (I i) := by
  /-
    α : Type u_2
    M : Matroid α
    ι : Type u_4
    A : Set ι
    hA : A.Nonempty
    I : ι → Set α
    h : M.Indep (Set.iUnion fun i => Set.iUnion fun h => I i)
    ⊢ Eq (M.closure (Set.iInter fun i => Set.iInter fun h => I i)) (Set.iInter fun …
  -/
  have := hA.coe_sort
  /-
    α : Type u_2
    M : Matroid α
    ι : Type u_4
    A : Set ι
    hA : A.Nonempty
    I : ι → Set α
    h : M.Indep (Set.iUnion fun i => Set.iUnion fun h => I i)
    this : Nonempty ↑A
    ⊢ Eq (M.closure (Set.iInter fun i => Set.iInter fun h => I i)) (Set.iInter fun …
  -/
  convert closure_iInter_eq_iInter_closure_of_iUnion_indep (Is := fun i : A ↦ I i) (by simpa) <;>
  /-
    case h.e'_2.h.e'_3
    α : Type u_2
    M : Matroid α
    ι : Type u_4
    A : Set ι
    hA : A.Nonempty
    I : ι → Set α
    h : M.Indep (Set.iUnion fun i => Set.iUnion fun h => I i)
    this : Nonempty ↑A
    ⊢ Eq (Set.iInter fun i => Set.iInter fun h => I i) (Set.iInter fun i => I ↑i)
  -/
  /-
    🎉 no goals
  -/
  simp
  /-
    🎉 no goals
  -/


lemma Indep.closure_iInter_eq_biInter_closure_of_forall_subset [Nonempty ι] {Js : ι → Set α}
    (hI : M.Indep I) (hJs : ∀ i, Js i ⊆ I) : M.closure (⋂ i, Js i) = ⋂ i, M.closure (Js i) :=
                                                                      /-
                                                                        α : Type u_2
                                                                        M : Matroid α
                                                                        ι : Sort u_3
                                                                        I : Set α
                                                                        inst✝ : Nonempty ι
                                                                        Js : ι → Set α
                                                                        hI : M.Indep I
                                                                        hJs : ∀ (i : ι), HasSubset.Subset (Js i) I
                                                                        ⊢ HasSubset.Subset (Set.iUnion fun i => Js i) I
                                                                      -/
  closure_iInter_eq_iInter_closure_of_iUnion_indep _ (hI.subset <| by simpa)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma Indep.closure_inter_eq_inter_closure (h : M.Indep (I ∪ J)) :
    M.closure (I ∩ J) = M.closure I ∩ M.closure J := by
  /-
    α : Type u_2
    M : Matroid α
    I J : Set α
    h : M.Indep (Union.union I J)
    ⊢ Eq (M.closure (Inter.inter I J)) (Inter.inter (M.closure I) (M.closure J))
  -/
  rw [inter_eq_iInter, closure_iInter_eq_iInter_closure_of_iUnion_indep, inter_eq_iInter]
    /-
      α : Type u_2
      M : Matroid α
      I J : Set α
      h : M.Indep (Union.union I J)
      ⊢ Eq (Set.iInter fun i => M.closure (cond i I J)) (Set.iInter fun b => cond b  …
    -/
  · exact iInter_congr (by simp)
    /-
      🎉 no goals
    -/
  /-
    case h
    α : Type u_2
    M : Matroid α
    I J : Set α
    h : M.Indep (Union.union I J)
    ⊢ M.Indep (Set.iUnion fun i => cond i I J)
  -/
  rwa [← union_eq_iUnion]
  /-
    🎉 no goals
  -/


lemma basis_iff_basis_closure_of_subset (hIX : I ⊆ X) (hX : X ⊆ M.E := by aesop_mat) :
    M.Basis I X ↔ M.Basis I (M.closure X) :=
  ⟨fun h ↦ h.basis_closure_right, fun h ↦ h.basis_subset hIX (M.subset_closure X hX)⟩


lemma basis_iff_basis_closure_of_subset' (hIX : I ⊆ X) :
    M.Basis I X ↔ M.Basis I (M.closure X) ∧ X ⊆ M.E :=
  ⟨fun h ↦ ⟨h.basis_closure_right, h.subset_ground⟩,
    fun h ↦ h.1.basis_subset hIX (M.subset_closure X h.2)⟩


lemma basis'_iff_basis_closure : M.Basis' I X ↔ M.Basis I (M.closure X) ∧ I ⊆ X := by
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    ⊢ Iff (M.Basis' I X) (And (M.Basis I (M.closure X)) (HasSubset.Subset I X))
  -/
  rw [← closure_inter_ground, basis'_iff_basis_inter_ground]
  exact ⟨fun h ↦ ⟨h.basis_closure_right, h.subset.trans inter_subset_left⟩,
    fun h ↦ h.1.basis_subset (subset_inter h.2 h.1.indep.subset_ground) (M.subset_closure _)⟩


lemma exists_basis_inter_ground_basis_closure (M : Matroid α) (X : Set α) :
    ∃ I, M.Basis I (X ∩ M.E) ∧ M.Basis I (M.closure X) := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    ⊢ Exists fun I => And (M.Basis I (Inter.inter X M.E)) (M.Basis I (M.closure X))
  -/
  obtain ⟨I, hI⟩ := M.exists_basis (X ∩ M.E)
  /-
    case intro
    α : Type u_2
    M : Matroid α
    X I : Set α
    hI : M.Basis I (Inter.inter X M.E)
    ⊢ Exists fun I => And (M.Basis I (Inter.inter X M.E)) (M.Basis I (M.closure X))
  -/
  have hI' := hI.basis_closure_right; rw [closure_inter_ground] at hI'
  /-
    case intro
    α : Type u_2
    M : Matroid α
    X I : Set α
    hI : M.Basis I (Inter.inter X M.E)
    hI' : M.Basis I (M.closure X)
    ⊢ Exists fun I => And (M.Basis I (Inter.inter X M.E)) (M.Basis I (M.closure X))
  -/
  exact ⟨_, hI, hI'⟩
  /-
    🎉 no goals
  -/


lemma Basis.basis_of_closure_eq_closure (hI : M.Basis I X) (hY : I ⊆ Y)
    (h : M.closure X = M.closure Y) (hYE : Y ⊆ M.E := by aesop_mat) : M.Basis I Y := by
  /-
    α : Type u_2
    M : Matroid α
    X Y I : Set α
    hI : M.Basis I X
    hY : HasSubset.Subset I Y
    h : Eq (M.closure X) (M.closure Y)
    hYE : autoParam (HasSubset.Subset Y M.E) _auto✝
    ⊢ M.Basis I Y
  -/
  refine hI.indep.basis_of_subset_of_subset_closure hY ?_
  /-
    α : Type u_2
    M : Matroid α
    X Y I : Set α
    hI : M.Basis I X
    hY : HasSubset.Subset I Y
    h : Eq (M.closure X) (M.closure Y)
    hYE : autoParam (HasSubset.Subset Y M.E) _auto✝
    ⊢ HasSubset.Subset Y (M.closure I)
  -/
  rw [hI.closure_eq_closure, h]
  /-
    α : Type u_2
    M : Matroid α
    X Y I : Set α
    hI : M.Basis I X
    hY : HasSubset.Subset I Y
    h : Eq (M.closure X) (M.closure Y)
    hYE : autoParam (HasSubset.Subset Y M.E) _auto✝
    ⊢ HasSubset.Subset Y (M.closure Y)
  -/
  exact M.subset_closure Y
  /-
    🎉 no goals
  -/


lemma basis_union_iff_indep_closure : M.Basis I (I ∪ X) ↔ M.Indep I ∧ X ⊆ M.closure I :=
  ⟨fun h ↦ ⟨h.indep, subset_union_right.trans h.subset_closure⟩, fun ⟨hI, hXI⟩ ↦
                                                                   /-
                                                                     α : Type u_2
                                                                     M : Matroid α
                                                                     X I : Set α
                                                                     x✝ : And (M.Indep I) (HasSubset.Subset X (M.closure I))
                                                                     hI : M.Indep I
                                                                     hXI : HasSubset.Subset X (M.closure I)
                                                                     ⊢ HasSubset.Subset I M.E
                                                                   -/
    hI.basis_closure.basis_subset subset_union_left (union_subset (M.subset_closure I) hXI)⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma basis_iff_indep_closure : M.Basis I X ↔ M.Indep I ∧ X ⊆ M.closure I ∧ I ⊆ X :=
  ⟨fun h ↦ ⟨h.indep, h.subset_closure, h.subset⟩, fun h ↦
    (basis_union_iff_indep_closure.mpr ⟨h.1, h.2.1⟩).basis_subset h.2.2 subset_union_right⟩


lemma Basis.eq_of_closure_subset (hI : M.Basis I X) (hJI : J ⊆ I) (hJ : X ⊆ M.closure J) :
    J = I := by
  /-
    α : Type u_2
    M : Matroid α
    X I J : Set α
    hI : M.Basis I X
    hJI : HasSubset.Subset J I
    hJ : HasSubset.Subset X (M.closure J)
    ⊢ Eq J I
  -/
  rw [← hI.indep.closure_inter_eq_self_of_subset hJI, inter_eq_self_of_subset_right]
  /-
    α : Type u_2
    M : Matroid α
    X I J : Set α
    hI : M.Basis I X
    hJI : HasSubset.Subset J I
    hJ : HasSubset.Subset X (M.closure J)
    ⊢ HasSubset.Subset I (M.closure J)
  -/
  exact hI.subset.trans hJ
  /-
    🎉 no goals
  -/


@[simp] lemma empty_basis_iff : M.Basis ∅ X ↔ X ⊆ M.closure ∅ := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    ⊢ Iff (M.Basis EmptyCollection.emptyCollection X) (HasSubset.Subset X (M.closu …
  -/
  rw [basis_iff_indep_closure, and_iff_right M.empty_indep, and_iff_left (empty_subset _)]
  /-
    🎉 no goals
  -/


lemma indep_iff_forall_not_mem_closure_diff (hI : I ⊆ M.E := by aesop_mat) :
    M.Indep I ↔ ∀ ⦃e⦄, e ∈ I → e ∉ M.closure (I \ {e}) := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    ⊢ Iff (M.Indep I) (∀ ⦃e : α⦄, Membership.mem I e → Not (Membership.mem (M.clos …
  -/
  use fun h e heI he ↦ ((h.closure_inter_eq_self_of_subset diff_subset).subset ⟨he, heI⟩).2 rfl
  /-
    case mpr
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    ⊢ (∀ ⦃e : α⦄, Membership.mem I e → Not (Membership.mem (M.closure (SDiff.sdiff …
  -/
  intro h
  /-
    case mpr
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    h : ∀ ⦃e : α⦄, Membership.mem I e → Not (Membership.mem (M.closure (SDiff.sdif …
    ⊢ M.Indep I
  -/
  obtain ⟨J, hJ⟩ := M.exists_basis I
  /-
    case mpr.intro
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    h : ∀ ⦃e : α⦄, Membership.mem I e → Not (Membership.mem (M.closure (SDiff.sdif …
    J : Set α
    hJ : M.Basis J I
    ⊢ M.Indep I
  -/
  convert hJ.indep
  /-
    case h.e'_3
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    h : ∀ ⦃e : α⦄, Membership.mem I e → Not (Membership.mem (M.closure (SDiff.sdif …
    J : Set α
    hJ : M.Basis J I
    ⊢ Eq I J
  -/
  refine hJ.subset.antisymm' (fun e he ↦ by_contra fun heJ ↦ h he ?_)
  exact mem_of_mem_of_subset
    (hJ.subset_closure he) (M.closure_subset_closure (subset_diff_singleton hJ.subset heJ))


/-- An alternative version of `Matroid.indep_iff_forall_not_mem_closure_diff` where the
hypothesis that `I ⊆ M.E` is contained in the RHS rather than the hypothesis. -/
lemma indep_iff_forall_not_mem_closure_diff' :
    M.Indep I ↔ I ⊆ M.E ∧ ∀ e ∈ I, e ∉ M.closure (I \ {e}) :=
  ⟨fun h ↦ ⟨h.subset_ground, (indep_iff_forall_not_mem_closure_diff h.subset_ground).mp h⟩, fun h ↦
    (indep_iff_forall_not_mem_closure_diff h.1).mpr h.2⟩


lemma Indep.not_mem_closure_diff_of_mem (hI : M.Indep I) (he : e ∈ I) : e ∉ M.closure (I \ {e}) :=
  (indep_iff_forall_not_mem_closure_diff'.1 hI).2 e he


lemma indep_iff_forall_closure_diff_ne :
    M.Indep I ↔ ∀ ⦃e⦄, e ∈ I → M.closure (I \ {e}) ≠ M.closure I := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    ⊢ Iff (M.Indep I) (∀ ⦃e : α⦄, Membership.mem I e → Ne (M.closure (SDiff.sdiff  …
  -/
  rw [indep_iff_forall_not_mem_closure_diff']
  refine ⟨fun ⟨hIE, h⟩ e heI h_eq ↦ h e heI (h_eq.symm.subset (M.mem_closure_of_mem heI)),
    fun h ↦ ⟨fun e heI ↦ by_contra fun heE ↦ h heI ?_,fun e heI hin ↦ h heI ?_⟩⟩
  · rw [← closure_inter_ground, inter_comm, inter_diff_distrib_left,
      inter_singleton_eq_empty.mpr heE, diff_empty, inter_comm, closure_inter_ground]
  /-
    case refine_2
    α : Type u_2
    M : Matroid α
    I : Set α
    h : ∀ ⦃e : α⦄, Membership.mem I e → Ne (M.closure (SDiff.sdiff I (Singleton.si …
    e : α
    heI : Membership.mem I e
    hin : Membership.mem (M.closure (SDiff.sdiff I (Singleton.singleton e))) e
    ⊢ Eq (M.closure (SDiff.sdiff I (Singleton.singleton e))) (M.closure I)
  -/
  nth_rw 2 [show I = insert e (I \ {e}) by simp [heI]]
  /-
    case refine_2
    α : Type u_2
    M : Matroid α
    I : Set α
    h : ∀ ⦃e : α⦄, Membership.mem I e → Ne (M.closure (SDiff.sdiff I (Singleton.si …
    e : α
    heI : Membership.mem I e
    hin : Membership.mem (M.closure (SDiff.sdiff I (Singleton.singleton e))) e
    ⊢ Eq (M.closure (SDiff.sdiff I (Singleton.singleton e))) (M.closure (Insert.in …
  -/
  rw [← closure_insert_closure_eq_closure_insert, insert_eq_of_mem hin, closure_closure]
  /-
    🎉 no goals
  -/


lemma Indep.closure_ssubset_closure (hI : M.Indep I) (hJI : J ⊂ I) : M.closure J ⊂ M.closure I := by
  /-
    α : Type u_2
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJI : HasSSubset.SSubset J I
    ⊢ HasSSubset.SSubset (M.closure J) (M.closure I)
  -/
  obtain ⟨e, heI, heJ⟩ := exists_of_ssubset hJI
  exact (M.closure_subset_closure hJI.subset).ssubset_of_not_subset fun hss ↦ heJ <|
    (hI.closure_inter_eq_self_of_subset hJI.subset).subset ⟨hss (M.mem_closure_of_mem heI), heI⟩


lemma indep_iff_forall_closure_ssubset_of_ssubset (hI : I ⊆ M.E := by aesop_mat) :
    M.Indep I ↔ ∀ ⦃J⦄, J ⊂ I → M.closure J ⊂ M.closure I := by
  refine ⟨fun h _ ↦ h.closure_ssubset_closure,
    fun h ↦ (indep_iff_forall_not_mem_closure_diff hI).2 fun e heI hecl ↦ ?_⟩
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    h : ∀ ⦃J : Set α⦄, HasSSubset.SSubset J I → HasSSubset.SSubset (M.closure J) ( …
    e : α
    heI : Membership.mem I e
    hecl : Membership.mem (M.closure (SDiff.sdiff I (Singleton.singleton e))) e
    ⊢ False
  -/
  refine (h (diff_singleton_sSubset.2 heI)).ne ?_
  rw [show I = insert e (I \ {e}) by simp [heI], ← closure_insert_closure_eq_closure_insert,
    insert_eq_of_mem hecl]
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    h : ∀ ⦃J : Set α⦄, HasSSubset.SSubset J I → HasSSubset.SSubset (M.closure J) ( …
    e : α
    heI : Membership.mem I e
    hecl : Membership.mem (M.closure (SDiff.sdiff I (Singleton.singleton e))) e
    ⊢ Eq (M.closure (SDiff.sdiff (Insert.insert e (SDiff.sdiff I (Singleton.single …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma Indep.closure_diff_ssubset (hI : M.Indep I) (hX : (I ∩ X).Nonempty) :
    M.closure (I \ X) ⊂ M.closure I := by
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    hI : M.Indep I
    hX : (Inter.inter I X).Nonempty
    ⊢ HasSSubset.SSubset (M.closure (SDiff.sdiff I X)) (M.closure I)
  -/
  refine hI.closure_ssubset_closure <| diff_subset.ssubset_of_ne fun h ↦ ?_
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    hI : M.Indep I
    hX : (Inter.inter I X).Nonempty
    h : Eq (SDiff.sdiff I X) I
    ⊢ False
  -/
  rw [sdiff_eq_left, disjoint_iff_inter_eq_empty] at h
  /-
    α : Type u_2
    M : Matroid α
    X I : Set α
    hI : M.Indep I
    hX : (Inter.inter I X).Nonempty
    h : Eq (Inter.inter I X) EmptyCollection.emptyCollection
    ⊢ False
  -/
  simp [h] at hX
  /-
    🎉 no goals
  -/


lemma Indep.closure_diff_singleton_ssubset (hI : M.Indep I) (he : e ∈ I) :
    M.closure (I \ {e}) ⊂ M.closure I :=
                                   /-
                                     α : Type u_2
                                     M : Matroid α
                                     e : α
                                     I : Set α
                                     hI : M.Indep I
                                     he : Membership.mem I e
                                     ⊢ HasSSubset.SSubset (SDiff.sdiff I (Singleton.singleton e)) I
                                   -/
  hI.closure_ssubset_closure <| by simpa
                                   /-
                                     🎉 no goals
                                   -/


lemma mem_closure_insert (he : e ∉ M.closure X) (hef : e ∈ M.closure (insert f X)) :
    f ∈ M.closure (insert e X) := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    he : Not (Membership.mem (M.closure X) e)
    hef : Membership.mem (M.closure (Insert.insert f X)) e
    ⊢ Membership.mem (M.closure (Insert.insert e X)) f
  -/
  rw [← closure_inter_ground] at *
  have hfE : f ∈ M.E := by
    by_contra! hfE; rw [insert_inter_of_not_mem hfE] at hef; exact he hef
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    he : Not (Membership.mem (M.closure (Inter.inter X M.E)) e)
    hef : Membership.mem (M.closure (Inter.inter (Insert.insert f X) M.E)) e
    hfE : Membership.mem M.E f
    ⊢ Membership.mem (M.closure (Inter.inter (Insert.insert e X) M.E)) f
  -/
  have heE : e ∈ M.E := (M.closure_subset_ground _) hef
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    he : Not (Membership.mem (M.closure (Inter.inter X M.E)) e)
    hef : Membership.mem (M.closure (Inter.inter (Insert.insert f X) M.E)) e
    hfE : Membership.mem M.E f
    heE : Membership.mem M.E e
    ⊢ Membership.mem (M.closure (Inter.inter (Insert.insert e X) M.E)) f
  -/
  rw [insert_inter_of_mem hfE] at hef; rw [insert_inter_of_mem heE]

  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    he : Not (Membership.mem (M.closure (Inter.inter X M.E)) e)
    hef : Membership.mem (M.closure (Insert.insert f (Inter.inter X M.E))) e
    hfE : Membership.mem M.E f
    heE : Membership.mem M.E e
    ⊢ Membership.mem (M.closure (Insert.insert e (Inter.inter X M.E))) f
  -/
  obtain ⟨I, hI⟩ := M.exists_basis (X ∩ M.E)
  /-
    case intro
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    he : Not (Membership.mem (M.closure (Inter.inter X M.E)) e)
    hef : Membership.mem (M.closure (Insert.insert f (Inter.inter X M.E))) e
    hfE : Membership.mem M.E f
    heE : Membership.mem M.E e
    I : Set α
    hI : M.Basis I (Inter.inter X M.E)
    ⊢ Membership.mem (M.closure (Insert.insert e (Inter.inter X M.E))) f
  -/
  rw [← hI.closure_eq_closure, hI.indep.not_mem_closure_iff] at he
  rw [← closure_insert_closure_eq_closure_insert, ← hI.closure_eq_closure,
    closure_insert_closure_eq_closure_insert, he.1.mem_closure_iff] at *
  rw [or_iff_not_imp_left, dep_iff, insert_comm,
    and_iff_left (insert_subset heE (insert_subset hfE hI.indep.subset_ground)), not_not]
  /-
    case intro
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    hfE : Membership.mem M.E f
    heE : Membership.mem M.E e
    I : Set α
    hef : Membership.mem (M.closure (Insert.insert f I)) e
    he : And (M.Indep (Insert.insert e I)) (Not (Membership.mem I e))
    hI : M.Basis I (Inter.inter X M.E)
    ⊢ M.Indep (Insert.insert e (Insert.insert f I)) → Membership.mem (Insert.inser …
  -/
  intro h
  rw [(h.subset (subset_insert _ _)).mem_closure_iff, or_iff_right (h.not_dep), mem_insert_iff,
    or_iff_left he.2] at hef
  /-
    case intro
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    hfE : Membership.mem M.E f
    heE : Membership.mem M.E e
    I : Set α
    hef : Eq e f
    he : And (M.Indep (Insert.insert e I)) (Not (Membership.mem I e))
    hI : M.Basis I (Inter.inter X M.E)
    h : M.Indep (Insert.insert e (Insert.insert f I))
    ⊢ Membership.mem (Insert.insert e I) f
  -/
  subst hef; apply mem_insert
             /-
               🎉 no goals
             -/


lemma closure_exchange (he : e ∈ M.closure (insert f X) \ M.closure X) :
    f ∈ M.closure (insert e X) \ M.closure X :=
  ⟨mem_closure_insert he.2 he.1, fun hf ↦ by
    /-
      α : Type u_2
      M : Matroid α
      X : Set α
      e f : α
      he : Membership.mem (SDiff.sdiff (M.closure (Insert.insert f X)) (M.closure X) …
      hf : Membership.mem (M.closure X) f
      ⊢ False
    -/
    rwa [closure_insert_eq_of_mem_closure hf, diff_self, iff_false_intro (not_mem_empty _)] at he⟩
    /-
      🎉 no goals
    -/


lemma closure_exchange_iff :
    e ∈ M.closure (insert f X) \ M.closure X ↔ f ∈ M.closure (insert e X) \ M.closure X :=
  ⟨closure_exchange, closure_exchange⟩


lemma closure_insert_congr (he : e ∈ M.closure (insert f X) \ M.closure X) :
    M.closure (insert e X) = M.closure (insert f X) := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e f : α
    he : Membership.mem (SDiff.sdiff (M.closure (Insert.insert f X)) (M.closure X) …
    ⊢ Eq (M.closure (Insert.insert e X)) (M.closure (Insert.insert f X))
  -/
  have hf := closure_exchange he
  rw [eq_comm, ← closure_closure, ← insert_eq_of_mem he.1, closure_insert_closure_eq_closure_insert,
    insert_comm, ← closure_closure, ← closure_insert_closure_eq_closure_insert,
    insert_eq_of_mem hf.1, closure_closure, closure_closure]


lemma closure_diff_eq_self (h : Y ⊆ M.closure (X \ Y)) : M.closure (X \ Y) = M.closure X := by
  rw [← diff_union_inter X Y, ← closure_union_closure_left_eq,
    union_eq_self_of_subset_right (inter_subset_right.trans h), closure_closure, diff_union_inter]


lemma closure_diff_singleton_eq_closure (h : e ∈ M.closure (X \ {e})) :
    M.closure (X \ {e}) = M.closure X :=
                           /-
                             α : Type u_2
                             M : Matroid α
                             X : Set α
                             e : α
                             h : Membership.mem (M.closure (SDiff.sdiff X (Singleton.singleton e))) e
                             ⊢ HasSubset.Subset (Singleton.singleton e) (M.closure (SDiff.sdiff X (Singleto …
                           -/
  closure_diff_eq_self (by simpa)
                           /-
                             🎉 no goals
                           -/


lemma subset_closure_diff_iff_closure_eq (h : Y ⊆ X) (hY : Y ⊆ M.E := by aesop_mat) :
    Y ⊆ M.closure (X \ Y) ↔ M.closure (X \ Y) = M.closure X :=
                                   /-
                                     α : Type u_2
                                     M : Matroid α
                                     X Y : Set α
                                     h : HasSubset.Subset Y X
                                     hY : autoParam (HasSubset.Subset Y M.E) _auto✝
                                     h' : Eq (M.closure (SDiff.sdiff X Y)) (M.closure X)
                                     ⊢ HasSubset.Subset Y M.E
                                   -/
  ⟨closure_diff_eq_self, fun h' ↦ (M.subset_closure_of_subset' h).trans h'.symm.subset⟩
                                   /-
                                     🎉 no goals
                                   -/


lemma mem_closure_diff_singleton_iff_closure (he : e ∈ X) (heE : e ∈ M.E := by aesop_mat) :
    e ∈ M.closure (X \ {e}) ↔ M.closure (X \ {e}) = M.closure X := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    e : α
    he : Membership.mem X e
    heE : autoParam (Membership.mem M.E e) _auto✝
    ⊢ Iff (Membership.mem (M.closure (SDiff.sdiff X (Singleton.singleton e))) e) ( …
  -/
  simpa using subset_closure_diff_iff_closure_eq (Y := {e}) (X := X) (by simpa)
  /-
    🎉 no goals
  -/


lemma ext_closure {M₁ M₂ : Matroid α} (h : ∀ X, M₁.closure X = M₂.closure X) : M₁ = M₂ :=
                /-
                  α : Type u_2
                  M₁ M₂ : Matroid α
                  h : ∀ (X : Set α), Eq (M₁.closure X) (M₂.closure X)
                  ⊢ Eq M₁.E M₂.E
                -/
  ext_indep (by simpa using h univ)
                /-
                  🎉 no goals
                -/
                  /-
                    α : Type u_2
                    M₁ M₂ : Matroid α
                    h : ∀ (X : Set α), Eq (M₁.closure X) (M₂.closure X)
                    x✝¹ : Set α
                    x✝ : HasSubset.Subset x✝¹ M₁.E
                    ⊢ Iff (M₁.Indep x✝¹) (M₂.Indep x✝¹)
                  -/
    (fun _ _ ↦ by simp_rw [indep_iff_forall_closure_diff_ne, h])
                  /-
                    🎉 no goals
                  -/



/-- A set is `spanning` in `M` if its closure is equal to `M.E`, or equivalently if it contains
  a base of `M`. -/
@[mk_iff]
structure Spanning (M : Matroid α) (S : Set α) : Prop where
  closure_eq : M.closure S = M.E
  subset_ground : S ⊆ M.E


lemma spanning_iff_closure_eq (hS : S ⊆ M.E := by aesop_mat) :
    M.Spanning S ↔ M.closure S = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : autoParam (HasSubset.Subset S M.E) _auto✝
    ⊢ Iff (M.Spanning S) (Eq (M.closure S) M.E)
  -/
  rw [spanning_iff, and_iff_left hS]
  /-
    🎉 no goals
  -/


@[simp] lemma closure_spanning_iff (hS : S ⊆ M.E := by aesop_mat) :
    M.Spanning (M.closure S) ↔ M.Spanning S := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : autoParam (HasSubset.Subset S M.E) _auto✝
    ⊢ Iff (M.Spanning (M.closure S)) (M.Spanning S)
  -/
  rw [spanning_iff_closure_eq, closure_closure, ← spanning_iff_closure_eq]
  /-
    🎉 no goals
  -/


lemma spanning_iff_ground_subset_closure (hS : S ⊆ M.E := by aesop_mat) :
    M.Spanning S ↔ M.E ⊆ M.closure S := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : autoParam (HasSubset.Subset S M.E) _auto✝
    ⊢ Iff (M.Spanning S) (HasSubset.Subset M.E (M.closure S))
  -/
  rw [spanning_iff_closure_eq, subset_antisymm_iff, and_iff_right (closure_subset_ground _ _)]
  /-
    🎉 no goals
  -/


lemma not_spanning_iff_closure_ssubset (hS : S ⊆ M.E := by aesop_mat) :
    ¬M.Spanning S ↔ M.closure S ⊂ M.E := by
  rw [spanning_iff_closure_eq, ssubset_iff_subset_ne, iff_and_self,
    iff_true_intro (M.closure_subset_ground _)]
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : autoParam (HasSubset.Subset S M.E) _auto✝
    ⊢ Not (Eq (M.closure S) M.E) → True
  -/
  exact fun _ ↦ trivial
  /-
    🎉 no goals
  -/


lemma Spanning.superset (hS : M.Spanning S) (hST : S ⊆ T) (hT : T ⊆ M.E := by aesop_mat) :
    M.Spanning T :=
  ⟨(M.closure_subset_ground _).antisymm
        /-
          α : Type u_2
          M : Matroid α
          S T : Set α
          hS : M.Spanning S
          hST : HasSubset.Subset S T
          hT : autoParam (HasSubset.Subset T M.E) _auto✝
          ⊢ HasSubset.Subset M.E (M.closure T)
        -/
    (by rw [← hS.closure_eq]; exact M.closure_subset_closure hST), hT⟩
                              /-
                                🎉 no goals
                              -/


lemma Spanning.closure_eq_of_superset (hS : M.Spanning S) (hST : S ⊆ T) : M.closure T = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    S T : Set α
    hS : M.Spanning S
    hST : HasSubset.Subset S T
    ⊢ Eq (M.closure T) M.E
  -/
  rw [← closure_inter_ground, ← spanning_iff_closure_eq]
  /-
    α : Type u_2
    M : Matroid α
    S T : Set α
    hS : M.Spanning S
    hST : HasSubset.Subset S T
    ⊢ M.Spanning (Inter.inter T M.E)
  -/
  exact hS.superset (subset_inter hST hS.subset_ground)
  /-
    🎉 no goals
  -/


lemma Spanning.union_left (hS : M.Spanning S) (hX : X ⊆ M.E := by aesop_mat) : M.Spanning (S ∪ X) :=
  /-
    α : Type u_2
    M : Matroid α
    X S : Set α
    hS : M.Spanning S
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ HasSubset.Subset (Union.union S X) M.E
  -/
  hS.superset subset_union_left
  /-
    🎉 no goals
  -/


lemma Spanning.union_right (hS : M.Spanning S) (hX : X ⊆ M.E := by aesop_mat) :
    M.Spanning (X ∪ S) :=
  /-
    α : Type u_2
    M : Matroid α
    X S : Set α
    hS : M.Spanning S
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ HasSubset.Subset (Union.union X S) M.E
  -/
  hS.superset subset_union_right
  /-
    🎉 no goals
  -/


lemma Base.spanning (hB : M.Base B) : M.Spanning B :=
  ⟨hB.closure_eq, hB.subset_ground⟩


lemma ground_spanning (M : Matroid α) : M.Spanning M.E :=
  ⟨M.closure_ground, rfl.subset⟩


lemma Base.spanning_of_superset (hB : M.Base B) (hBX : B ⊆ X) (hX : X ⊆ M.E := by aesop_mat) :
    M.Spanning X :=
  /-
    α : Type u_2
    M : Matroid α
    X B : Set α
    hB : M.Base B
    hBX : HasSubset.Subset B X
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ HasSubset.Subset X M.E
  -/
  hB.spanning.superset hBX
  /-
    🎉 no goals
  -/


/-- A version of `Matroid.spanning_iff_exists_base_subset` in which the `S ⊆ M.E` condition
appears in the RHS of the equivalence rather than as a hypothesis. -/
lemma spanning_iff_exists_base_subset' : M.Spanning S ↔ (∃ B, M.Base B ∧ B ⊆ S) ∧ S ⊆ M.E := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    ⊢ Iff (M.Spanning S) (And (Exists fun B => And (M.Base B) (HasSubset.Subset B  …
  -/
  refine ⟨fun h ↦ ⟨?_, h.subset_ground⟩, fun ⟨⟨B, hB, hBS⟩, hSE⟩ ↦ hB.spanning.superset hBS⟩
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    h : M.Spanning S
    ⊢ Exists fun B => And (M.Base B) (HasSubset.Subset B S)
  -/
  obtain ⟨B, hB⟩ := M.exists_basis S
  /-
    case intro
    α : Type u_2
    M : Matroid α
    S : Set α
    h : M.Spanning S
    B : Set α
    hB : M.Basis B S
    ⊢ Exists fun B => And (M.Base B) (HasSubset.Subset B S)
  -/
  have hB' := hB.basis_closure_right
  /-
    case intro
    α : Type u_2
    M : Matroid α
    S : Set α
    h : M.Spanning S
    B : Set α
    hB : M.Basis B S
    hB' : M.Basis B (M.closure S)
    ⊢ Exists fun B => And (M.Base B) (HasSubset.Subset B S)
  -/
  rw [h.closure_eq, basis_ground_iff] at hB'
  /-
    case intro
    α : Type u_2
    M : Matroid α
    S : Set α
    h : M.Spanning S
    B : Set α
    hB : M.Basis B S
    hB' : M.Base B
    ⊢ Exists fun B => And (M.Base B) (HasSubset.Subset B S)
  -/
  exact ⟨B, hB', hB.subset⟩
  /-
    🎉 no goals
  -/


lemma spanning_iff_exists_base_subset (hS : S ⊆ M.E := by aesop_mat) :
    M.Spanning S ↔ ∃ B, M.Base B ∧ B ⊆ S := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : autoParam (HasSubset.Subset S M.E) _auto✝
    ⊢ Iff (M.Spanning S) (Exists fun B => And (M.Base B) (HasSubset.Subset B S))
  -/
  rw [spanning_iff_exists_base_subset', and_iff_left hS]
  /-
    🎉 no goals
  -/


lemma Spanning.exists_base_subset (hS : M.Spanning S) : ∃ B, M.Base B ∧ B ⊆ S := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : M.Spanning S
    ⊢ Exists fun B => And (M.Base B) (HasSubset.Subset B S)
  -/
  rwa [spanning_iff_exists_base_subset] at hS
  /-
    🎉 no goals
  -/


lemma coindep_iff_compl_spanning (hI : I ⊆ M.E := by aesop_mat) :
    M.Coindep I ↔ M.Spanning (M.E \ I) := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : autoParam (HasSubset.Subset I M.E) _auto✝
    ⊢ Iff (M.Coindep I) (M.Spanning (SDiff.sdiff M.E I))
  -/
  rw [coindep_iff_exists, spanning_iff_exists_base_subset]
  /-
    🎉 no goals
  -/


lemma spanning_iff_compl_coindep (hS : S ⊆ M.E := by aesop_mat) :
    M.Spanning S ↔ M.Coindep (M.E \ S) := by
  /-
    α : Type u_2
    M : Matroid α
    S : Set α
    hS : autoParam (HasSubset.Subset S M.E) _auto✝
    ⊢ Iff (M.Spanning S) (M.Coindep (SDiff.sdiff M.E S))
  -/
  rw [coindep_iff_compl_spanning, diff_diff_cancel_left hS]
  /-
    🎉 no goals
  -/


lemma Coindep.compl_spanning (hI : M.Coindep I) : M.Spanning (M.E \ I) :=
  (coindep_iff_compl_spanning hI.subset_ground).mp hI


lemma coindep_iff_closure_compl_eq_ground (hK : X ⊆ M.E := by aesop_mat) :
    M.Coindep X ↔ M.closure (M.E \ X) = M.E := by
  /-
    α : Type u_2
    M : Matroid α
    X : Set α
    hK : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (M.Coindep X) (Eq (M.closure (SDiff.sdiff M.E X)) M.E)
  -/
  rw [coindep_iff_compl_spanning, spanning_iff_closure_eq]
  /-
    🎉 no goals
  -/


lemma Coindep.closure_compl (hX : M.Coindep X) : M.closure (M.E \ X) = M.E :=
  (coindep_iff_closure_compl_eq_ground hX.subset_ground).mp hX


lemma Indep.base_of_spanning (hI : M.Indep I) (hIs : M.Spanning I) : M.Base I := by
  /-
    α : Type u_2
    M : Matroid α
    I : Set α
    hI : M.Indep I
    hIs : M.Spanning I
    ⊢ M.Base I
  -/
  obtain ⟨B, hB, hBI⟩ := hIs.exists_base_subset; rwa [← hB.eq_of_subset_indep hI hBI]
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma Spanning.base_of_indep (hIs : M.Spanning I) (hI : M.Indep I) : M.Base I :=
  hI.base_of_spanning hIs


lemma ext_spanning {M M' : Matroid α} (h : M.E = M'.E)
    (hsp : ∀ S, S ⊆ M.E → (M.Spanning S ↔ M'.Spanning S )) : M = M' := by
  have hsp' : M.Spanning = M'.Spanning := by
    ext S
    refine (em (S ⊆ M.E)).elim (fun hSE ↦ by rw [hsp _ hSE] )
      (fun hSE ↦ iff_of_false (fun h ↦ hSE h.subset_ground)
      (fun h' ↦ hSE (h'.subset_ground.trans h.symm.subset)))
  /-
    α : Type u_2
    M M' : Matroid α
    h : Eq M.E M'.E
    hsp : ∀ (S : Set α), HasSubset.Subset S M.E → Iff (M.Spanning S) (M'.Spanning S)
    hsp' : Eq M.Spanning M'.Spanning
    ⊢ Eq M M'
  -/
  rw [← dual_inj, ext_iff_indep, dual_ground, dual_ground, and_iff_right h]
  /-
    α : Type u_2
    M M' : Matroid α
    h : Eq M.E M'.E
    hsp : ∀ (S : Set α), HasSubset.Subset S M.E → Iff (M.Spanning S) (M'.Spanning S)
    hsp' : Eq M.Spanning M'.Spanning
    ⊢ ∀ ⦃I : Set α⦄, HasSubset.Subset I M.E → Iff (M.dual.Indep I) (M'.dual.Indep I)
  -/
  intro I hIE
  /-
    α : Type u_2
    M M' : Matroid α
    h : Eq M.E M'.E
    hsp : ∀ (S : Set α), HasSubset.Subset S M.E → Iff (M.Spanning S) (M'.Spanning S)
    hsp' : Eq M.Spanning M'.Spanning
    I : Set α
    hIE : HasSubset.Subset I M.E
    ⊢ Iff (M.dual.Indep I) (M'.dual.Indep I)
  -/
  rw [← coindep_def, ← coindep_def, coindep_iff_compl_spanning, coindep_iff_compl_spanning, hsp', h]
  /-
    🎉 no goals
  -/


