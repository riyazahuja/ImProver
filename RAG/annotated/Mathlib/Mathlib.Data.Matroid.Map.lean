/-- The pullback of a matroid on `β` by a function `f : α → β` to a matroid on `α`.
Elements with the same (nonloop) image are parallel and the ground set is `f ⁻¹' M.E`.
The matroids `M.comap f` and `M ↾ range f` have isomorphic simplifications;
the preimage of each nonloop of `M ↾ range f` is a parallel class. -/
def comap (N : Matroid β) (f : α → β) : Matroid α :=
  IndepMatroid.matroid <|
  { E := f ⁻¹' N.E
    Indep := fun I ↦ N.Indep (f '' I) ∧ InjOn f I
                      /-
                        α : Type u_1
                        β : Type u_2
                        f✝ : α → β
                        E I : Set α
                        M : Matroid α
                        N✝ N : Matroid β
                        f : α → β
                        ⊢ (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) EmptyCollection.emp …
                      -/
    indep_empty := by simp
                      /-
                        🎉 no goals
                      -/
    indep_subset := fun _ _ h hIJ ↦ ⟨h.1.subset (image_subset _ hIJ), InjOn.mono hIJ h.2⟩
    indep_aug := by
      /-
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        ⊢ ∀ ⦃I B : Set α⦄, (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) I  …
      -/
      rintro I B ⟨hI, hIinj⟩ hImax hBmax
      /-
        case intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        I B : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hImax : Not (Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) I)
        hBmax : Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) B
        ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) ((fun I => And (N.I …
      -/
      obtain ⟨I', hII', hI', hI'inj⟩ := (not_maximal_subset_iff ⟨hI, hIinj⟩).1 hImax

      have h₁ : ¬(N ↾ range f).Base (f '' I) := by
        refine fun hB ↦ hII'.ne ?_
        have h_im := hB.eq_of_subset_indep (by simpa) (image_subset _ hII'.subset)
        rwa [hI'inj.image_eq_image_iff hII'.subset Subset.rfl] at h_im

      have h₂ : (N ↾ range f).Base (f '' B) := by
        refine Indep.base_of_forall_insert (by simpa using hBmax.1.1) ?_
        rintro _ ⟨⟨e, heB, rfl⟩, hfe⟩ hi
        rw [restrict_indep_iff, ← image_insert_eq] at hi
        have hinj : InjOn f (insert e B) := by
          rw [injOn_insert (fun heB ↦ hfe (mem_image_of_mem f heB))]
          exact ⟨hBmax.1.2, hfe⟩
        refine hBmax.not_prop_of_ssuperset (t := insert e B) (ssubset_insert ?_) ⟨hi.1, hinj⟩
        exact fun heB ↦ hfe <| mem_image_of_mem f heB

      /-
        case intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        I B : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hImax : Not (Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) I)
        hBmax : Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) B
        I' : Set α
        hII' : HasSSubset.SSubset I I'
        hI' : N.Indep (Set.image f I')
        hI'inj : Set.InjOn f I'
        h₁ : Not ((N.restrict (Set.range f)).Base (Set.image f I))
        h₂ : (N.restrict (Set.range f)).Base (Set.image f B)
        ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) ((fun I => And (N.I …
      -/
      obtain ⟨_, ⟨⟨e, he, rfl⟩, he'⟩, hei⟩ := Indep.exists_insert_of_not_base (by simpa) h₁ h₂
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        I B : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hImax : Not (Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) I)
        hBmax : Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) B
        I' : Set α
        hII' : HasSSubset.SSubset I I'
        hI' : N.Indep (Set.image f I')
        hI'inj : Set.InjOn f I'
        h₁ : Not ((N.restrict (Set.range f)).Base (Set.image f I))
        h₂ : (N.restrict (Set.range f)).Base (Set.image f B)
        e : α
        he : Membership.mem B e
        hei : (N.restrict (Set.range f)).Indep (Insert.insert (f e) (Set.image f I))
        he' : Not (Membership.mem (Set.image f I) (f e))
        ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) ((fun I => And (N.I …
      -/
      have heI : e ∉ I := fun heI ↦ he' (mem_image_of_mem f heI)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        I B : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hImax : Not (Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) I)
        hBmax : Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) B
        I' : Set α
        hII' : HasSSubset.SSubset I I'
        hI' : N.Indep (Set.image f I')
        hI'inj : Set.InjOn f I'
        h₁ : Not ((N.restrict (Set.range f)).Base (Set.image f I))
        h₂ : (N.restrict (Set.range f)).Base (Set.image f B)
        e : α
        he : Membership.mem B e
        hei : (N.restrict (Set.range f)).Indep (Insert.insert (f e) (Set.image f I))
        he' : Not (Membership.mem (Set.image f I) (f e))
        heI : Not (Membership.mem I e)
        ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) ((fun I => And (N.I …
      -/
      rw [← image_insert_eq, restrict_indep_iff] at hei
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        I B : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hImax : Not (Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) I)
        hBmax : Maximal (fun I => And (N.Indep (Set.image f I)) (Set.InjOn f I)) B
        I' : Set α
        hII' : HasSSubset.SSubset I I'
        hI' : N.Indep (Set.image f I')
        hI'inj : Set.InjOn f I'
        h₁ : Not ((N.restrict (Set.range f)).Base (Set.image f I))
        h₂ : (N.restrict (Set.range f)).Base (Set.image f B)
        e : α
        he : Membership.mem B e
        hei : And (N.Indep (Set.image f (Insert.insert e I))) (HasSubset.Subset (Set.i …
        he' : Not (Membership.mem (Set.image f I) (f e))
        heI : Not (Membership.mem I e)
        ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) ((fun I => And (N.I …
      -/
      exact ⟨e, ⟨he, heI⟩, hei.1, (injOn_insert heI).2 ⟨hIinj, he'⟩⟩
      /-
        🎉 no goals
      -/

    indep_maximal := by
      /-
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        ⊢ ∀ (X : Set α), HasSubset.Subset X (Set.preimage f N.E) → Matroid.ExistsMaxim …
      -/
      rintro X - I ⟨hI, hIinj⟩ hIX
      obtain ⟨J, hJ⟩ := (N ↾ range f).existsMaximalSubsetProperty_indep (f '' X) (by simp)
        (f '' I) (by simpa) (image_subset _ hIX)

      simp only [restrict_indep_iff, image_subset_iff, maximal_subset_iff, mem_setOf_eq, and_imp,
        and_assoc] at hJ ⊢

      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        X I : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hIX : HasSubset.Subset I X
        J : Set β
        hJ : And (HasSubset.Subset I (Set.preimage f J)) (And (N.Indep J) (And (HasSub …
        ⊢ Exists fun J => And (HasSubset.Subset I J) (And (N.Indep (Set.image f J)) (A …
      -/
      obtain ⟨hIJ, hJ, hJf, hJX, hJmax⟩ := hJ
      obtain ⟨J₀, hIJ₀, hJ₀X, hbj⟩ := hIinj.bijOn_image.exists_extend_of_subset hIX
        (image_subset f hIJ) (image_subset_iff.2 <| preimage_mono hJX)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        X I : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hIX : HasSubset.Subset I X
        J : Set β
        hIJ : HasSubset.Subset I (Set.preimage f J)
        hJ : N.Indep J
        hJf : HasSubset.Subset J (Set.range f)
        hJX : HasSubset.Subset J (Set.image f X)
        hJmax : ∀ ⦃t : Set β⦄, N.Indep t → HasSubset.Subset t (Set.range f) → HasSubse …
        J₀ : Set α
        hIJ₀ : HasSubset.Subset I J₀
        hJ₀X : HasSubset.Subset J₀ X
        hbj : Set.BijOn f J₀ (Set.image f (Set.preimage f J))
        ⊢ Exists fun J => And (HasSubset.Subset I J) (And (N.Indep (Set.image f J)) (A …
      -/
      obtain rfl : f '' J₀ = J := by rw [← image_preimage_eq_of_subset hJf, hbj.image_eq]
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I✝ : Set α
        M : Matroid α
        N✝ N : Matroid β
        f : α → β
        X I : Set α
        hI : N.Indep (Set.image f I)
        hIinj : Set.InjOn f I
        hIX : HasSubset.Subset I X
        J₀ : Set α
        hIJ₀ : HasSubset.Subset I J₀
        hJ₀X : HasSubset.Subset J₀ X
        hIJ : HasSubset.Subset I (Set.preimage f (Set.image f J₀))
        hJ : N.Indep (Set.image f J₀)
        hJf : HasSubset.Subset (Set.image f J₀) (Set.range f)
        hJX : HasSubset.Subset (Set.image f J₀) (Set.image f X)
        hJmax : ∀ ⦃t : Set β⦄, N.Indep t → HasSubset.Subset t (Set.range f) → HasSubse …
        hbj : Set.BijOn f J₀ (Set.image f (Set.preimage f (Set.image f J₀)))
        ⊢ Exists fun J => And (HasSubset.Subset I J) (And (N.Indep (Set.image f J)) (A …
      -/
      refine ⟨J₀, hIJ₀, hJ, hbj.injOn, hJ₀X, fun K hK hKinj hKX hJ₀K ↦ ?_⟩
      rw [← hKinj.image_eq_image_iff hJ₀K Subset.rfl, hJmax hK (image_subset_range _ _)
        (image_subset f hKX) (image_subset f hJ₀K)]
    subset_ground := fun _ hI e heI  ↦ hI.1.subset_ground ⟨e, heI, rfl⟩ }


@[simp] lemma comap_indep_iff : (N.comap f).Indep I ↔ N.Indep (f '' I) ∧ InjOn f I := Iff.rfl


@[simp] lemma comap_ground_eq (N : Matroid β) (f : α → β) : (N.comap f).E = f ⁻¹' N.E := rfl


@[simp] lemma comap_dep_iff :
    (N.comap f).Dep I ↔ N.Dep (f '' I) ∨ (N.Indep (f '' I) ∧ ¬ InjOn f I) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    ⊢ Iff ((N.comap f).Dep I) (Or (N.Dep (Set.image f I)) (And (N.Indep (Set.image …
  -/
  rw [Dep, comap_indep_iff, not_and, comap_ground_eq, Dep, image_subset_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    ⊢ Iff (And (N.Indep (Set.image f I) → Not (Set.InjOn f I)) (HasSubset.Subset I …
  -/
  refine ⟨fun ⟨hi, h⟩ ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      I : Set α
      N : Matroid β
      x✝ : And (N.Indep (Set.image f I) → Not (Set.InjOn f I)) (HasSubset.Subset I ( …
      hi : N.Indep (Set.image f I) → Not (Set.InjOn f I)
      h : HasSubset.Subset I (Set.preimage f N.E)
      ⊢ Or (And (Not (N.Indep (Set.image f I))) (HasSubset.Subset I (Set.preimage f  …
    -/
  · rw [and_iff_left h, ← imp_iff_not_or]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      I : Set α
      N : Matroid β
      x✝ : And (N.Indep (Set.image f I) → Not (Set.InjOn f I)) (HasSubset.Subset I ( …
      hi : N.Indep (Set.image f I) → Not (Set.InjOn f I)
      h : HasSubset.Subset I (Set.preimage f N.E)
      ⊢ N.Indep (Set.image f I) → And (N.Indep (Set.image f I)) (Not (Set.InjOn f I))
    -/
    exact fun hI ↦ ⟨hI, hi hI⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    ⊢ Or (And (Not (N.Indep (Set.image f I))) (HasSubset.Subset I (Set.preimage f  …
  -/
  rintro (⟨hI, hIE⟩ | hI)
    /-
      case refine_2.inl.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      I : Set α
      N : Matroid β
      hI : Not (N.Indep (Set.image f I))
      hIE : HasSubset.Subset I (Set.preimage f N.E)
      ⊢ And (N.Indep (Set.image f I) → Not (Set.InjOn f I)) (HasSubset.Subset I (Set …
    -/
  · exact ⟨fun h ↦ (hI h).elim, hIE⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2.inr
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    hI : And (N.Indep (Set.image f I)) (Not (Set.InjOn f I))
    ⊢ And (N.Indep (Set.image f I) → Not (Set.InjOn f I)) (HasSubset.Subset I (Set …
  -/
  rw [iff_true_intro hI.1, iff_true_intro hI.2, implies_true, true_and]
  /-
    case refine_2.inr
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    hI : And (N.Indep (Set.image f I)) (Not (Set.InjOn f I))
    ⊢ HasSubset.Subset I (Set.preimage f N.E)
  -/
  simpa using hI.1.subset_ground
  /-
    🎉 no goals
  -/


@[simp] lemma comap_id (N : Matroid β) : N.comap id = N :=
                      /-
                        β : Type u_2
                        N : Matroid β
                        ⊢ ∀ ⦃I : Set β⦄, HasSubset.Subset I (N.comap id).E → Iff ((N.comap id).Indep I …
                      -/
  ext_indep rfl <| by simp [injective_id.injOn]
                      /-
                        🎉 no goals
                      -/


lemma comap_indep_iff_of_injOn (hf : InjOn f (f ⁻¹' N.E)) :
    (N.comap f).Indep I ↔ N.Indep (f '' I) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    hf : Set.InjOn f (Set.preimage f N.E)
    ⊢ Iff ((N.comap f).Indep I) (N.Indep (Set.image f I))
  -/
  rw [comap_indep_iff, and_iff_left_iff_imp]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    hf : Set.InjOn f (Set.preimage f N.E)
    ⊢ N.Indep (Set.image f I) → Set.InjOn f I
  -/
  refine fun hi ↦ hf.mono <| subset_trans ?_ (preimage_mono hi.subset_ground)
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    I : Set α
    N : Matroid β
    hf : Set.InjOn f (Set.preimage f N.E)
    hi : N.Indep (Set.image f I)
    ⊢ HasSubset.Subset I (Set.preimage f (Set.image f I))
  -/
  apply subset_preimage_image
  /-
    🎉 no goals
  -/


@[simp] lemma comap_emptyOn (f : α → β) : comap (emptyOn β) f = emptyOn α := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Eq ((Matroid.emptyOn β).comap f) (Matroid.emptyOn α)
  -/
  simp [← ground_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma comap_loopyOn (f : α → β) (E : Set β) : comap (loopyOn E) f = loopyOn (f ⁻¹' E) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    E : Set β
    ⊢ Eq ((Matroid.loopyOn E).comap f) (Matroid.loopyOn (Set.preimage f E))
  -/
  rw [eq_loopyOn_iff]; aesop
                       /-
                         🎉 no goals
                       -/


@[simp] lemma comap_basis_iff {I X : Set α} :
    (N.comap f).Basis I X ↔ N.Basis (f '' I) (f '' X) ∧ I.InjOn f ∧ I ⊆ X  := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    I X : Set α
    ⊢ Iff ((N.comap f).Basis I X) (And (N.Basis (Set.image f I) (Set.image f X)) ( …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      h : (N.comap f).Basis I X
      ⊢ And (N.Basis (Set.image f I) (Set.image f X)) (And (Set.InjOn f I) (HasSubse …
    -/
  · obtain ⟨hI, hinj⟩ := comap_indep_iff.1 h.indep
    /-
      case refine_1.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      h : (N.comap f).Basis I X
      hI : N.Indep (Set.image f I)
      hinj : Set.InjOn f I
      ⊢ And (N.Basis (Set.image f I) (Set.image f X)) (And (Set.InjOn f I) (HasSubse …
    -/
    refine ⟨hI.basis_of_forall_insert (image_subset f h.subset) fun e he ↦ ?_, hinj, h.subset⟩
    simp only [mem_diff, mem_image, not_exists, not_and, and_imp, forall_exists_index,
      forall_apply_eq_imp_iff₂] at he
    /-
      case refine_1.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      h : (N.comap f).Basis I X
      hI : N.Indep (Set.image f I)
      hinj : Set.InjOn f I
      e : β
      he : And (Exists fun x => And (Membership.mem X x) (Eq (f x) e)) (∀ (x : α), M …
      ⊢ N.Dep (Insert.insert e (Set.image f I))
    -/
    obtain ⟨⟨e, heX, rfl⟩, he⟩ := he
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      h : (N.comap f).Basis I X
      hI : N.Indep (Set.image f I)
      hinj : Set.InjOn f I
      e : α
      heX : Membership.mem X e
      he : ∀ (x : α), Membership.mem I x → Not (Eq (f x) (f e))
      ⊢ N.Dep (Insert.insert (f e) (Set.image f I))
    -/
    have heI : e ∉ I := fun heI ↦ (he e heI rfl)
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      h : (N.comap f).Basis I X
      hI : N.Indep (Set.image f I)
      hinj : Set.InjOn f I
      e : α
      heX : Membership.mem X e
      he : ∀ (x : α), Membership.mem I x → Not (Eq (f x) (f e))
      heI : Not (Membership.mem I e)
      ⊢ N.Dep (Insert.insert (f e) (Set.image f I))
    -/
    replace h := h.insert_dep ⟨heX, heI⟩
    simp only [comap_dep_iff, image_insert_eq, or_iff_not_imp_right, injOn_insert heI,
      hinj, mem_image, not_exists, not_and, true_and, not_forall, Classical.not_imp, not_not] at h
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      hI : N.Indep (Set.image f I)
      hinj : Set.InjOn f I
      e : α
      heX : Membership.mem X e
      he : ∀ (x : α), Membership.mem I x → Not (Eq (f x) (f e))
      heI : Not (Membership.mem I e)
      h : (N.Indep (Insert.insert (f e) (Set.image f I)) → ∀ (x : α), Membership.mem …
      ⊢ N.Dep (Insert.insert (f e) (Set.image f I))
    -/
    exact h (fun _ ↦ he)
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    I X : Set α
    h : And (N.Basis (Set.image f I) (Set.image f X)) (And (Set.InjOn f I) (HasSub …
    ⊢ (N.comap f).Basis I X
  -/
  refine Indep.basis_of_forall_insert ?_ h.2.2 fun e ⟨heX, heI⟩ ↦ ?_
    /-
      case refine_2.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      N : Matroid β
      I X : Set α
      h : And (N.Basis (Set.image f I) (Set.image f X)) (And (Set.InjOn f I) (HasSub …
      ⊢ (N.comap f).Indep I
    -/
  · simp [comap_indep_iff, h.1.indep, h.2]
    /-
      🎉 no goals
    -/
  have hIE : insert e I ⊆ (N.comap f).E := by
      simp_rw [comap_ground_eq, ← image_subset_iff]
      exact (image_subset _ (insert_subset heX h.2.2)).trans h.1.subset_ground
  suffices N.Indep (insert (f e) (f '' I)) → ∃ x ∈ I, f x = f e
    by simpa [← not_indep_iff hIE, injOn_insert heI, h.2.1, image_insert_eq]
  /-
    case refine_2.refine_2
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    I X : Set α
    h : And (N.Basis (Set.image f I) (Set.image f X)) (And (Set.InjOn f I) (HasSub …
    e : α
    x✝ : Membership.mem (SDiff.sdiff X I) e
    heX : Membership.mem X e
    heI : Not (Membership.mem I e)
    hIE : HasSubset.Subset (Insert.insert e I) (N.comap f).E
    ⊢ N.Indep (Insert.insert (f e) (Set.image f I)) → Exists fun x => And (Members …
  -/
  exact h.1.mem_of_insert_indep (mem_image_of_mem f heX)
  /-
    🎉 no goals
  -/


@[simp] lemma comap_base_iff {B : Set α} :
    (N.comap f).Base B ↔ N.Basis (f '' B) (f '' (f ⁻¹' N.E)) ∧ B.InjOn f ∧ B ⊆ f ⁻¹' N.E := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    B : Set α
    ⊢ Iff ((N.comap f).Base B) (And (N.Basis (Set.image f B) (Set.image f (Set.pre …
  -/
  rw [← basis_ground_iff, comap_basis_iff]; rfl
                                            /-
                                              🎉 no goals
                                            -/


@[simp] lemma comap_basis'_iff {I X : Set α} :
    (N.comap f).Basis' I X ↔ N.Basis' (f '' I) (f '' X) ∧ I.InjOn f ∧ I ⊆ X := by
  simp only [basis'_iff_basis_inter_ground, comap_ground_eq, comap_basis_iff, image_inter_preimage,
    subset_inter_iff, ← and_assoc, and_congr_left_iff, and_iff_left_iff_imp, and_imp]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    I X : Set α
    ⊢ N.Basis (Set.image f I) (Inter.inter (Set.image f X) N.E) → Set.InjOn f I →  …
  -/
  exact fun h _ _ ↦ (image_subset_iff.1 h.indep.subset_ground)
  /-
    🎉 no goals
  -/


instance comap_finitary (N : Matroid β) [N.Finitary] (f : α → β) : (N.comap f).Finitary := by
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.Finitary
    f : α → β
    ⊢ (N.comap f).Finitary
  -/
  refine ⟨fun I hI ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.Finitary
    f : α → β
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → (N.comap f).Indep J
    ⊢ (N.comap f).Indep I
  -/
  rw [comap_indep_iff, indep_iff_forall_finite_subset_indep]
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.Finitary
    f : α → β
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → (N.comap f).Indep J
    ⊢ And (∀ (J : Set β), HasSubset.Subset J (Set.image f I) → J.Finite → N.Indep  …
  -/
  simp only [forall_subset_image_iff]
  refine ⟨fun J hJ hfin ↦ ?_,
    fun x hx y hy ↦ (hI _ (pair_subset hx hy) (by simp)).2 (by simp) (by simp)⟩
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.Finitary
    f : α → β
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → (N.comap f).Indep J
    J : Set α
    hJ : HasSubset.Subset J I
    hfin : (Set.image f J).Finite
    ⊢ N.Indep (Set.image f J)
  -/
  obtain ⟨J', hJ'J, hJ'⟩ := (surjOn_image f J).exists_bijOn_subset
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.Finitary
    f : α → β
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → (N.comap f).Indep J
    J : Set α
    hJ : HasSubset.Subset J I
    hfin : (Set.image f J).Finite
    J' : Set α
    hJ'J : HasSubset.Subset J' J
    hJ' : Set.BijOn f J' (Set.image f J)
    ⊢ N.Indep (Set.image f J)
  -/
  rw [← hJ'.image_eq] at hfin ⊢
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.Finitary
    f : α → β
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → (N.comap f).Indep J
    J : Set α
    hJ : HasSubset.Subset J I
    J' : Set α
    hfin : (Set.image f J').Finite
    hJ'J : HasSubset.Subset J' J
    hJ' : Set.BijOn f J' (Set.image f J)
    ⊢ N.Indep (Set.image f J')
  -/
  exact (hI J' (hJ'J.trans hJ) (hfin.of_finite_image hJ'.injOn)).1
  /-
    🎉 no goals
  -/


instance comap_finiteRk (N : Matroid β) [N.FiniteRk] (f : α → β) : (N.comap f).FiniteRk := by
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.FiniteRk
    f : α → β
    ⊢ (N.comap f).FiniteRk
  -/
  obtain ⟨B, hB⟩ := (N.comap f).exists_base
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.FiniteRk
    f : α → β
    B : Set α
    hB : (N.comap f).Base B
    ⊢ (N.comap f).FiniteRk
  -/
  refine hB.finiteRk_of_finite ?_
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.FiniteRk
    f : α → β
    B : Set α
    hB : (N.comap f).Base B
    ⊢ B.Finite
  -/
  simp only [comap_base_iff] at hB
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N✝ N : Matroid β
    inst✝ : N.FiniteRk
    f : α → β
    B : Set α
    hB : And (N.Basis (Set.image f B) (Set.image f (Set.preimage f N.E))) (And (Se …
    ⊢ B.Finite
  -/
  exact (hB.1.indep.finite.of_finite_image hB.2.1)
  /-
    🎉 no goals
  -/


/-- The pullback of a matroid on `β` by a function `f : α → β` to a matroid on `α`,
restricted to a ground set `E`.
The matroids `M.comapOn f E` and `M ↾ (f '' E)` have isomorphic simplifications;
elements with the same nonloop image are parallel. -/
def comapOn (N : Matroid β) (E : Set α) (f : α → β) : Matroid α := (N.comap f) ↾ E


lemma comapOn_preimage_eq (N : Matroid β) (f : α → β) : N.comapOn (f ⁻¹' N.E) f = N.comap f := by
  /-
    α : Type u_1
    β : Type u_2
    N : Matroid β
    f : α → β
    ⊢ Eq (N.comapOn (Set.preimage f N.E) f) (N.comap f)
  -/
  rw [comapOn, restrict_eq_self_iff]; rfl
                                      /-
                                        🎉 no goals
                                      -/


@[simp] lemma comapOn_indep_iff :
    (N.comapOn E f).Indep I ↔ (N.Indep (f '' I) ∧ InjOn f I ∧ I ⊆ E) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    E I : Set α
    ⊢ Iff ((N.comapOn E f).Indep I) (And (N.Indep (Set.image f I)) (And (Set.InjOn …
  -/
  simp [comapOn, and_assoc]
  /-
    🎉 no goals
  -/


@[simp] lemma comapOn_ground_eq : (N.comapOn E f).E = E := rfl


lemma comapOn_base_iff :
    (N.comapOn E f).Base B ↔ N.Basis' (f '' B) (f '' E) ∧ B.InjOn f ∧ B ⊆ E := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    E B : Set α
    ⊢ Iff ((N.comapOn E f).Base B) (And (N.Basis' (Set.image f B) (Set.image f E)) …
  -/
  rw [comapOn, base_restrict_iff', comap_basis'_iff]
  /-
    🎉 no goals
  -/


lemma comapOn_base_iff_of_surjOn (h : SurjOn f E N.E) :
    (N.comapOn E f).Base B ↔ (N.Base (f '' B) ∧ InjOn f B ∧ B ⊆ E) := by
  simp_rw [comapOn_base_iff, and_congr_left_iff, and_imp,
    basis'_iff_basis_inter_ground, inter_eq_self_of_subset_right h, basis_ground_iff, implies_true]


lemma comapOn_base_iff_of_bijOn (h : BijOn f E N.E) :
    (N.comapOn E f).Base B ↔ N.Base (f '' B) ∧ B ⊆ E := by
  rw [← and_iff_left_of_imp (Base.subset_ground (M := N.comapOn E f) (B := B)),
    comapOn_ground_eq, and_congr_left_iff]
  suffices h' : B ⊆ E → InjOn f B from fun hB ↦
    by simp [hB, comapOn_base_iff_of_surjOn h.surjOn, h']
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    E B : Set α
    h : Set.BijOn f E N.E
    ⊢ HasSubset.Subset B E → Set.InjOn f B
  -/
  exact fun hBE ↦ h.injOn.mono hBE
  /-
    🎉 no goals
  -/


lemma comapOn_dual_eq_of_bijOn (h : BijOn f E N.E) :
    (N.comapOn E f)✶ = N✶.comapOn E f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    E : Set α
    h : Set.BijOn f E N.E
    ⊢ Eq (N.comapOn E f).dual (N.dual.comapOn E f)
  -/
  refine ext_base (by simp) (fun B hB ↦ ?_)
  rw [comapOn_base_iff_of_bijOn (by simpa), dual_base_iff, comapOn_base_iff_of_bijOn h,
    dual_base_iff _, comapOn_ground_eq, and_iff_left diff_subset, and_iff_left (by simpa),
    h.injOn.image_diff_subset (by simpa), h.image_eq]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    N : Matroid β
    E : Set α
    h : Set.BijOn f E N.E
    B : Set α
    hB : HasSubset.Subset B (N.comapOn E f).dual.E
    ⊢ HasSubset.Subset (Set.image f B) N.E
  -/
  exact (h.mapsTo.mono_left (show B ⊆ E by simpa)).image_subset
  /-
    🎉 no goals
  -/


instance comapOn_finitary [N.Finitary] : (N.comapOn E f).Finitary := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M : Matroid α
    N : Matroid β
    E B I : Set α
    inst✝ : N.Finitary
    ⊢ (N.comapOn E f).Finitary
  -/
  rw [comapOn]; infer_instance
                /-
                  🎉 no goals
                -/


instance comapOn_finiteRk [N.FiniteRk] : (N.comapOn E f).FiniteRk := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M : Matroid α
    N : Matroid β
    E B I : Set α
    inst✝ : N.FiniteRk
    ⊢ (N.comapOn E f).FiniteRk
  -/
  rw [comapOn]; infer_instance
                /-
                  🎉 no goals
                -/


/-- Map a matroid `M` to an isomorphic copy in `β` using an embedding `M.E ↪ β`. -/
def mapSetEmbedding (M : Matroid α) (f : M.E ↪ β) : Matroid β := Matroid.ofExistsMatroid
  (E := range f)
  (Indep := fun I ↦ M.Indep ↑(f ⁻¹' I) ∧ I ⊆ range f)
  (hM := by
    classical
    obtain (rfl | ⟨⟨e,he⟩⟩) := eq_emptyOn_or_nonempty M
    · refine ⟨emptyOn β, ?_⟩
      simp only [emptyOn_ground] at f
      simp [range_eq_empty f, subset_empty_iff]
    have _ : Nonempty M.E := ⟨⟨e,he⟩⟩
    have _ : Nonempty α := ⟨e⟩
    refine ⟨M.comapOn (range f) (fun x ↦ ↑(invFunOn f univ x)), rfl, ?_⟩
    simp_rw [comapOn_indep_iff, ← and_assoc, and_congr_left_iff, subset_range_iff_exists_image_eq]
    rintro _ ⟨I, rfl⟩
    rw [← image_image, InjOn.invFunOn_image f.injective.injOn (subset_univ _),
      preimage_image_eq _ f.injective, and_iff_left_iff_imp]
    rintro - x hx y hy
    simp only [EmbeddingLike.apply_eq_iff_eq, Subtype.val_inj]
    exact (invFunOn_injOn_image f univ) (image_subset f (subset_univ I) hx)
      (image_subset f (subset_univ I) hy) )


@[simp] lemma mapSetEmbedding_ground (M : Matroid α) (f : M.E ↪ β) :
    (M.mapSetEmbedding f).E = range f := rfl


@[simp] lemma mapSetEmbedding_indep_iff {f : M.E ↪ β} {I : Set β} :
    (M.mapSetEmbedding f).Indep I ↔ M.Indep ↑(f ⁻¹' I) ∧ I ⊆ range f := Iff.rfl


lemma Indep.exists_eq_image_of_mapSetEmbedding {f : M.E ↪ β} {I : Set β}
    (hI : (M.mapSetEmbedding f).Indep I) : ∃ (I₀ : Set M.E), M.Indep I₀ ∧ I = f '' I₀ :=
  ⟨f ⁻¹' I, hI.1, Eq.symm <| image_preimage_eq_of_subset hI.2⟩


lemma mapSetEmbedding_indep_iff' {f : M.E ↪ β} {I : Set β} :
    (M.mapSetEmbedding f).Indep I ↔ ∃ (I₀ : Set M.E), M.Indep ↑I₀ ∧ I = f '' I₀ := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding (↑M.E) β
    I : Set β
    ⊢ Iff ((M.mapSetEmbedding f).Indep I) (Exists fun I₀ => And (M.Indep (Set.imag …
  -/
  simp only [mapSetEmbedding_indep_iff, subset_range_iff_exists_image_eq]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding (↑M.E) β
    I : Set β
    ⊢ Iff (And (M.Indep (Set.image Subtype.val (Set.preimage (⇑f) I))) (Exists fun …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : Function.Embedding (↑M.E) β
      I : Set β
      ⊢ And (M.Indep (Set.image Subtype.val (Set.preimage (⇑f) I))) (Exists fun t => …
    -/
  · rintro ⟨hI, I, rfl⟩
    /-
      case mp.intro.intro
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : Function.Embedding (↑M.E) β
      I : Set ↑M.E
      hI : M.Indep (Set.image Subtype.val (Set.preimage (⇑f) (Set.image (⇑f) I)))
      ⊢ Exists fun I₀ => And (M.Indep (Set.image Subtype.val I₀)) (Eq (Set.image (⇑f …
    -/
    exact ⟨I, by rwa [preimage_image_eq _ f.injective] at hI, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding (↑M.E) β
    I : Set β
    ⊢ (Exists fun I₀ => And (M.Indep (Set.image Subtype.val I₀)) (Eq I (Set.image  …
  -/
  rintro ⟨I, hI, rfl⟩
  /-
    case mpr.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding (↑M.E) β
    I : Set ↑M.E
    hI : M.Indep (Set.image Subtype.val I)
    ⊢ And (M.Indep (Set.image Subtype.val (Set.preimage (⇑f) (Set.image (⇑f) I)))) …
  -/
  rw [preimage_image_eq _ f.injective]
  /-
    case mpr.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding (↑M.E) β
    I : Set ↑M.E
    hI : M.Indep (Set.image Subtype.val I)
    ⊢ And (M.Indep (Set.image Subtype.val I)) (Exists fun t => Eq (Set.image (⇑f)  …
  -/
  exact ⟨hI, _, rfl⟩
  /-
    🎉 no goals
  -/


/-- Given a function `f` that is injective on `M.E`, the copy of `M` in `β` whose independent sets
are the images of those in `M`. If `β` is a nonempty type, then `N : Matroid β` is a map of `M`
if and only if `M` and `N` are isomorphic. -/
def map (M : Matroid α) (f : α → β) (hf : InjOn f M.E) : Matroid β := Matroid.ofExistsMatroid
  (E := f '' M.E)
  (Indep := fun I ↦ ∃ I₀, M.Indep I₀ ∧ I = f '' I₀)
  (hM := by
    /-
      α : Type u_1
      β : Type u_2
      f✝ : α → β
      E I : Set α
      M✝ : Matroid α
      N : Matroid β
      M : Matroid α
      f : α → β
      hf : Set.InjOn f M.E
      ⊢ Exists fun M_1 => And (Eq (Set.image f M.E) M_1.E) (∀ (I : Set β), Iff (M_1. …
    -/
    refine ⟨M.mapSetEmbedding ⟨_, hf.injective⟩, by simp, fun I ↦ ?_⟩
    simp_rw [mapSetEmbedding_indep_iff', Embedding.coeFn_mk, restrict_apply,
      ← image_image f Subtype.val, Subtype.exists_set_subtype (p := fun J ↦ M.Indep J ∧ I = f '' J)]
    /-
      α : Type u_1
      β : Type u_2
      f✝ : α → β
      E I✝ : Set α
      M✝ : Matroid α
      N : Matroid β
      M : Matroid α
      f : α → β
      hf : Set.InjOn f M.E
      I : Set β
      ⊢ Iff (Exists fun s => And (HasSubset.Subset s M.E) (And (M.Indep s) (Eq I (Se …
    -/
    exact ⟨fun ⟨I₀, _, hI₀⟩ ↦ ⟨I₀, hI₀⟩, fun ⟨I₀, hI₀⟩ ↦ ⟨I₀, hI₀.1.subset_ground, hI₀⟩⟩)
    /-
      🎉 no goals
    -/


@[simp] lemma map_ground (M : Matroid α) (f : α → β) (hf) : (M.map f hf).E = f '' M.E := rfl


@[simp] lemma map_indep_iff {hf} {I : Set β} :
    (M.map f hf).Indep I ↔ ∃ I₀, M.Indep I₀ ∧ I = f '' I₀ := Iff.rfl


lemma Indep.map (hI : M.Indep I) (f : α → β) (hf) : (M.map f hf).Indep (f '' I) :=
  map_indep_iff.2 ⟨I, hI, rfl⟩


lemma Indep.exists_bijOn_of_map {I : Set β} (hf) (hI : (M.map f hf).Indep I) :
    ∃ I₀, M.Indep I₀ ∧ BijOn f I₀ I := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    I : Set β
    hf : Set.InjOn f M.E
    hI : (M.map f hf).Indep I
    ⊢ Exists fun I₀ => And (M.Indep I₀) (Set.BijOn f I₀ I)
  -/
  obtain ⟨I₀, hI₀, rfl⟩ := hI
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    I₀ : Set α
    hI₀ : M.Indep I₀
    ⊢ Exists fun I₀_1 => And (M.Indep I₀_1) (Set.BijOn f I₀_1 (Set.image f I₀))
  -/
  exact ⟨I₀, hI₀, (hf.mono hI₀.subset_ground).bijOn_image⟩
  /-
    🎉 no goals
  -/


lemma map_image_indep_iff {hf} {I : Set α} (hI : I ⊆ M.E) :
    (M.map f hf).Indep (f '' I) ↔ M.Indep I := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    I : Set α
    hI : HasSubset.Subset I M.E
    ⊢ Iff ((M.map f hf).Indep (Set.image f I)) (M.Indep I)
  -/
  rw [map_indep_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    I : Set α
    hI : HasSubset.Subset I M.E
    ⊢ Iff (Exists fun I₀ => And (M.Indep I₀) (Eq (Set.image f I) (Set.image f I₀)) …
  -/
  refine ⟨fun ⟨J, hJ, hIJ⟩ ↦ ?_, fun h ↦ ⟨I, h, rfl⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    I : Set α
    hI : HasSubset.Subset I M.E
    x✝ : Exists fun I₀ => And (M.Indep I₀) (Eq (Set.image f I) (Set.image f I₀))
    J : Set α
    hJ : M.Indep J
    hIJ : Eq (Set.image f I) (Set.image f J)
    ⊢ M.Indep I
  -/
  rw [hf.image_eq_image_iff hI hJ.subset_ground] at hIJ; rwa [hIJ]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp] lemma map_base_iff (M : Matroid α) (f : α → β) (hf) {B : Set β} :
    (M.map f hf).Base B ↔ ∃ B₀, M.Base B₀ ∧ B = f '' B₀ := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set β
    ⊢ Iff ((M.map f hf).Base B) (Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image …
  -/
  rw [base_iff_maximal_indep]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set β
    ⊢ Iff (Maximal (M.map f hf).Indep B) (Exists fun B₀ => And (M.Base B₀) (Eq B ( …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : α → β
      hf : Set.InjOn f M.E
      B : Set β
      h : Maximal (M.map f hf).Indep B
      ⊢ Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image f B₀))
    -/
  · obtain ⟨B₀, hB₀, hbij⟩ := h.prop.exists_bijOn_of_map
    /-
      case refine_1.intro.intro
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : α → β
      hf : Set.InjOn f M.E
      B : Set β
      h : Maximal (M.map f hf).Indep B
      B₀ : Set α
      hB₀ : M.Indep B₀
      hbij : Set.BijOn f B₀ B
      ⊢ Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image f B₀))
    -/
    refine ⟨B₀, hB₀.base_of_maximal fun J hJ hB₀J ↦ ?_, hbij.image_eq.symm⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : α → β
      hf : Set.InjOn f M.E
      B : Set β
      h : Maximal (M.map f hf).Indep B
      B₀ : Set α
      hB₀ : M.Indep B₀
      hbij : Set.BijOn f B₀ B
      J : Set α
      hJ : M.Indep J
      hB₀J : HasSubset.Subset B₀ J
      ⊢ Eq B₀ J
    -/
    rw [← hf.image_eq_image_iff hB₀.subset_ground hJ.subset_ground, hbij.image_eq]
    /-
      case refine_1.intro.intro
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : α → β
      hf : Set.InjOn f M.E
      B : Set β
      h : Maximal (M.map f hf).Indep B
      B₀ : Set α
      hB₀ : M.Indep B₀
      hbij : Set.BijOn f B₀ B
      J : Set α
      hJ : M.Indep J
      hB₀J : HasSubset.Subset B₀ J
      ⊢ Eq B (Set.image f J)
    -/
    exact h.eq_of_subset (hJ.map f hf) (hbij.image_eq ▸ image_subset f hB₀J)
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set β
    ⊢ (Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image f B₀))) → Maximal (M.map  …
  -/
  rintro ⟨B, hB, rfl⟩
  /-
    case refine_2.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set α
    hB : M.Base B
    ⊢ Maximal (M.map f hf).Indep (Set.image f B)
  -/
  rw [maximal_subset_iff]
  /-
    case refine_2.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set α
    hB : M.Base B
    ⊢ And ((M.map f hf).Indep (Set.image f B)) (∀ ⦃t : Set β⦄, (M.map f hf).Indep  …
  -/
  refine ⟨hB.indep.map f hf, fun I hI hBI ↦ ?_⟩
  /-
    case refine_2.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set α
    hB : M.Base B
    I : Set β
    hI : (M.map f hf).Indep I
    hBI : HasSubset.Subset (Set.image f B) I
    ⊢ Eq (Set.image f B) I
  -/
  obtain ⟨I₀, hI₀, hbij⟩ := hI.exists_bijOn_of_map
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set α
    hB : M.Base B
    I : Set β
    hI : (M.map f hf).Indep I
    hBI : HasSubset.Subset (Set.image f B) I
    I₀ : Set α
    hI₀ : M.Indep I₀
    hbij : Set.BijOn f I₀ I
    ⊢ Eq (Set.image f B) I
  -/
  rw [← hbij.image_eq, hf.image_subset_image_iff hB.subset_ground hI₀.subset_ground] at hBI
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : α → β
    hf : Set.InjOn f M.E
    B : Set α
    hB : M.Base B
    I : Set β
    hI : (M.map f hf).Indep I
    I₀ : Set α
    hBI : HasSubset.Subset B I₀
    hI₀ : M.Indep I₀
    hbij : Set.BijOn f I₀ I
    ⊢ Eq (Set.image f B) I
  -/
  rw [hB.eq_of_subset_indep hI₀ hBI, hbij.image_eq]
  /-
    🎉 no goals
  -/


lemma Base.map {B : Set α} (hB : M.Base B) {f : α → β} (hf) : (M.map f hf).Base (f '' B) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    B : Set α
    hB : M.Base B
    f : α → β
    hf : Set.InjOn f M.E
    ⊢ (M.map f hf).Base (Set.image f B)
  -/
  rw [map_base_iff]; exact ⟨B, hB, rfl⟩
                     /-
                       🎉 no goals
                     -/


lemma map_dep_iff {hf} {D : Set β} :
    (M.map f hf).Dep D ↔ ∃ D₀, M.Dep D₀ ∧ D = f '' D₀ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    D : Set β
    ⊢ Iff ((M.map f hf).Dep D) (Exists fun D₀ => And (M.Dep D₀) (Eq D (Set.image f …
  -/
  simp only [Dep, map_indep_iff, not_exists, not_and, map_ground, subset_image_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    D : Set β
    ⊢ Iff (And (∀ (x : Set α), M.Indep x → Not (Eq D (Set.image f x))) (Exists fun …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      f : α → β
      M : Matroid α
      hf : Set.InjOn f M.E
      D : Set β
      ⊢ And (∀ (x : Set α), M.Indep x → Not (Eq D (Set.image f x))) (Exists fun u => …
    -/
  · rintro ⟨h, D₀, hD₀E, rfl⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      M : Matroid α
      hf : Set.InjOn f M.E
      D₀ : Set α
      hD₀E : HasSubset.Subset D₀ M.E
      h : ∀ (x : Set α), M.Indep x → Not (Eq (Set.image f D₀) (Set.image f x))
      ⊢ Exists fun D₀_1 => And (And (Not (M.Indep D₀_1)) (HasSubset.Subset D₀_1 M.E) …
    -/
    exact ⟨D₀, ⟨fun hd ↦ h _ hd rfl, hD₀E⟩, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    D : Set β
    ⊢ (Exists fun D₀ => And (And (Not (M.Indep D₀)) (HasSubset.Subset D₀ M.E)) (Eq …
  -/
  rintro ⟨D₀, ⟨hD₀, hD₀E⟩, rfl⟩
  /-
    case mpr.intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    D₀ : Set α
    hD₀ : Not (M.Indep D₀)
    hD₀E : HasSubset.Subset D₀ M.E
    ⊢ And (∀ (x : Set α), M.Indep x → Not (Eq (Set.image f D₀) (Set.image f x))) ( …
  -/
  refine ⟨fun I hI h_eq ↦ ?_, ⟨_, hD₀E, rfl⟩⟩
  /-
    case mpr.intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    D₀ : Set α
    hD₀ : Not (M.Indep D₀)
    hD₀E : HasSubset.Subset D₀ M.E
    I : Set α
    hI : M.Indep I
    h_eq : Eq (Set.image f D₀) (Set.image f I)
    ⊢ False
  -/
  rw [hf.image_eq_image_iff hD₀E hI.subset_ground] at h_eq
  /-
    case mpr.intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    D₀ : Set α
    hD₀ : Not (M.Indep D₀)
    hD₀E : HasSubset.Subset D₀ M.E
    I : Set α
    hI : M.Indep I
    h_eq : Eq D₀ I
    ⊢ False
  -/
  subst h_eq; contradiction
              /-
                🎉 no goals
              -/


lemma map_image_base_iff {hf} {B : Set α} (hB : B ⊆ M.E) :
    (M.map f hf).Base (f '' B) ↔ M.Base B := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    B : Set α
    hB : HasSubset.Subset B M.E
    ⊢ Iff ((M.map f hf).Base (Set.image f B)) (M.Base B)
  -/
  rw [map_base_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    B : Set α
    hB : HasSubset.Subset B M.E
    ⊢ Iff (Exists fun B₀ => And (M.Base B₀) (Eq (Set.image f B) (Set.image f B₀))) …
  -/
  refine ⟨fun ⟨J, hJ, hIJ⟩ ↦ ?_, fun h ↦ ⟨B, h, rfl⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    B : Set α
    hB : HasSubset.Subset B M.E
    x✝ : Exists fun B₀ => And (M.Base B₀) (Eq (Set.image f B) (Set.image f B₀))
    J : Set α
    hJ : M.Base J
    hIJ : Eq (Set.image f B) (Set.image f J)
    ⊢ M.Base B
  -/
  rw [hf.image_eq_image_iff hB hJ.subset_ground] at hIJ; rwa [hIJ]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma Basis.map {X : Set α} (hIX : M.Basis I X) {f : α → β} (hf) :
    (M.map f hf).Basis (f '' I) (f '' X) := by
  /-
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    ⊢ (M.map f hf).Basis (Set.image f I) (Set.image f X)
  -/
  refine (hIX.indep.map f hf).basis_of_forall_insert (image_subset _ hIX.subset) ?_
  /-
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    ⊢ ∀ (e : β), Membership.mem (SDiff.sdiff (Set.image f X) (Set.image f I)) e →  …
  -/
  rintro _ ⟨⟨e,he,rfl⟩, he'⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    ⊢ (M.map f hf).Dep (Insert.insert (f e) (Set.image f I))
  -/
  have hss := insert_subset (hIX.subset_ground he) hIX.indep.subset_ground
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    hss : HasSubset.Subset (Insert.insert e I) M.E
    ⊢ (M.map f hf).Dep (Insert.insert (f e) (Set.image f I))
  -/
  rw [← not_indep_iff (by simpa [← image_insert_eq] using image_subset f hss)]
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    hss : HasSubset.Subset (Insert.insert e I) M.E
    ⊢ Not ((M.map f hf).Indep (Insert.insert (f e) (Set.image f I)))
  -/
  simp only [map_indep_iff, not_exists, not_and]
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    hss : HasSubset.Subset (Insert.insert e I) M.E
    ⊢ ∀ (x : Set α), M.Indep x → Not (Eq (Insert.insert (f e) (Set.image f I)) (Se …
  -/
  intro J hJ hins
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    hss : HasSubset.Subset (Insert.insert e I) M.E
    J : Set α
    hJ : M.Indep J
    hins : Eq (Insert.insert (f e) (Set.image f I)) (Set.image f J)
    ⊢ False
  -/
  rw [← image_insert_eq, hf.image_eq_image_iff hss hJ.subset_ground] at hins
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    hss : HasSubset.Subset (Insert.insert e I) M.E
    J : Set α
    hJ : M.Indep J
    hins : Eq (Insert.insert e I) J
    ⊢ False
  -/
  obtain rfl := hins
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : α → β
    hf : Set.InjOn f M.E
    e : α
    he : Membership.mem X e
    he' : Not (Membership.mem (Set.image f I) (f e))
    hss : HasSubset.Subset (Insert.insert e I) M.E
    hJ : M.Indep (Insert.insert e I)
    ⊢ False
  -/
  exact he' (mem_image_of_mem f (hIX.mem_of_insert_indep he hJ))
  /-
    🎉 no goals
  -/


lemma map_basis_iff {I X : Set α} (f : α → β) (hf) (hI : I ⊆ M.E) (hX : X ⊆ M.E) :
    (M.map f hf).Basis (f '' I) (f '' X) ↔ M.Basis I X := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    ⊢ Iff ((M.map f hf).Basis (Set.image f I) (Set.image f X)) (M.Basis I X)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.map hf⟩
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    ⊢ M.Basis I X
  -/
  obtain ⟨I', hI', hII'⟩ := map_indep_iff.1 h.indep
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    I' : Set α
    hI' : M.Indep I'
    hII' : Eq (Set.image f I) (Set.image f I')
    ⊢ M.Basis I X
  -/
  rw [hf.image_eq_image_iff hI hI'.subset_ground] at hII'
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    I' : Set α
    hI' : M.Indep I'
    hII' : Eq I I'
    ⊢ M.Basis I X
  -/
  obtain rfl := hII'
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    hI' : M.Indep I
    ⊢ M.Basis I X
  -/
  have hss := (hf.image_subset_image_iff hI hX).1 h.subset
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    hI' : M.Indep I
    hss : HasSubset.Subset I X
    ⊢ M.Basis I X
  -/
  refine hI'.basis_of_maximal_subset hss (fun J hJ hIJ hJX ↦ ?_)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    hI' : M.Indep I
    hss : HasSubset.Subset I X
    J : Set α
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    hJX : HasSubset.Subset J X
    ⊢ HasSubset.Subset J I
  -/
  have hIJ' := h.eq_of_subset_indep (hJ.map f hf) (image_subset f hIJ) (image_subset f hJX)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    hI' : M.Indep I
    hss : HasSubset.Subset I X
    J : Set α
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    hJX : HasSubset.Subset J X
    hIJ' : Eq (Set.image f I) (Set.image f J)
    ⊢ HasSubset.Subset J I
  -/
  rw [hf.image_eq_image_iff hI hJ.subset_ground] at hIJ'
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    I X : Set α
    f : α → β
    hf : Set.InjOn f M.E
    hI : HasSubset.Subset I M.E
    hX : HasSubset.Subset X M.E
    h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
    hI' : M.Indep I
    hss : HasSubset.Subset I X
    J : Set α
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    hJX : HasSubset.Subset J X
    hIJ' : Eq I J
    ⊢ HasSubset.Subset J I
  -/
  exact hIJ'.symm.subset
  /-
    🎉 no goals
  -/


lemma map_basis_iff' {I X : Set β} {hf} :
    (M.map f hf).Basis I X ↔ ∃ I₀ X₀, M.Basis I₀ X₀ ∧ I = f '' I₀ ∧ X = f '' X₀ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    I X : Set β
    hf : Set.InjOn f M.E
    ⊢ Iff ((M.map f hf).Basis I X) (Exists fun I₀ => Exists fun X₀ => And (M.Basis …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      M : Matroid α
      I X : Set β
      hf : Set.InjOn f M.E
      h : (M.map f hf).Basis I X
      ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.image  …
    -/
  · obtain ⟨I, hI, rfl⟩ := subset_image_iff.1 h.indep.subset_ground
    /-
      case refine_1.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      M : Matroid α
      X : Set β
      hf : Set.InjOn f M.E
      I : Set α
      hI : HasSubset.Subset I M.E
      h : (M.map f hf).Basis (Set.image f I) X
      ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq (Set.image f  …
    -/
    obtain ⟨X, hX, rfl⟩ := subset_image_iff.1 h.subset_ground
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      M : Matroid α
      hf : Set.InjOn f M.E
      I : Set α
      hI : HasSubset.Subset I M.E
      X : Set α
      hX : HasSubset.Subset X M.E
      h : (M.map f hf).Basis (Set.image f I) (Set.image f X)
      ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq (Set.image f  …
    -/
    rw [map_basis_iff _ _ hI hX] at h
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      M : Matroid α
      hf : Set.InjOn f M.E
      I : Set α
      hI : HasSubset.Subset I M.E
      X : Set α
      hX : HasSubset.Subset X M.E
      h : M.Basis I X
      ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq (Set.image f  …
    -/
    exact ⟨I, X, h, rfl, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    I X : Set β
    hf : Set.InjOn f M.E
    ⊢ (Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.image …
  -/
  rintro ⟨I, X, hIX, rfl, rfl⟩
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    I X : Set α
    hIX : M.Basis I X
    ⊢ (M.map f hf).Basis (Set.image f I) (Set.image f X)
  -/
  exact hIX.map hf
  /-
    🎉 no goals
  -/


@[simp] lemma map_dual {hf} : (M.map f hf)✶ = M✶.map f hf := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    ⊢ Eq (M.map f hf).dual (M.dual.map f hf)
  -/
  apply ext_base (by simp)
  simp only [dual_ground, map_ground, subset_image_iff, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, dual_base_iff']
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    ⊢ ∀ (a : Set α), HasSubset.Subset a M.E → Iff (And ((M.map f hf).Base (SDiff.s …
  -/
  intro B hB
  simp_rw [← hf.image_diff_subset hB, map_image_base_iff diff_subset,
    map_image_base_iff (show B ⊆ M✶.E from hB), dual_base_iff hB, and_iff_left_iff_imp]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    M : Matroid α
    hf : Set.InjOn f M.E
    B : Set α
    hB : HasSubset.Subset B M.E
    ⊢ M.Base (SDiff.sdiff M.E B) → Exists fun u => And (HasSubset.Subset u M.E) (E …
  -/
  exact fun _ ↦ ⟨B, hB, rfl⟩
  /-
    🎉 no goals
  -/


                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                f✝ : α → β
                                                                E I : Set α
                                                                M : Matroid α
                                                                N : Matroid β
                                                                f : α → β
                                                                ⊢ Set.InjOn f (Matroid.emptyOn α).E
                                                              -/
@[simp] lemma map_emptyOn (f : α → β) : (emptyOn α).map f (by simp) = emptyOn β := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Eq ((Matroid.emptyOn α).map f ⋯) (Matroid.emptyOn β)
  -/
  simp [← ground_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma map_loopyOn (f : α → β) (hf) : (loopyOn E).map f hf = loopyOn (f '' E) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Set α
    f : α → β
    hf : Set.InjOn f (Matroid.loopyOn E).E
    ⊢ Eq ((Matroid.loopyOn E).map f hf) (Matroid.loopyOn (Set.image f E))
  -/
  simp [eq_loopyOn_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma map_freeOn (f : α → β) (hf) : (freeOn E).map f hf = freeOn (f '' E) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Set α
    f : α → β
    hf : Set.InjOn f (Matroid.freeOn E).E
    ⊢ Eq ((Matroid.freeOn E).map f hf) (Matroid.freeOn (Set.image f E))
  -/
  rw [← dual_inj]; simp
                   /-
                     🎉 no goals
                   -/


@[simp] lemma map_id : M.map id (injOn_id M.E) = M := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq (M.map id ⋯) M
  -/
  simp [ext_iff_indep]
  /-
    🎉 no goals
  -/


lemma map_comap {f : α → β} (h_range : N.E ⊆ range f) (hf : InjOn f (f ⁻¹' N.E)) :
    (N.comap f).map f hf = N := by
  /-
    α : Type u_1
    β : Type u_2
    N : Matroid β
    f : α → β
    h_range : HasSubset.Subset N.E (Set.range f)
    hf : Set.InjOn f (Set.preimage f N.E)
    ⊢ Eq ((N.comap f).map f hf) N
  -/
  refine ext_indep (by simpa [image_preimage_eq_iff]) ?_
  /-
    α : Type u_1
    β : Type u_2
    N : Matroid β
    f : α → β
    h_range : HasSubset.Subset N.E (Set.range f)
    hf : Set.InjOn f (Set.preimage f N.E)
    ⊢ ∀ ⦃I : Set β⦄, HasSubset.Subset I ((N.comap f).map f hf).E → Iff (((N.comap  …
  -/
  simp only [map_ground, comap_ground_eq, map_indep_iff, comap_indep_iff, forall_subset_image_iff]
  /-
    α : Type u_1
    β : Type u_2
    N : Matroid β
    f : α → β
    h_range : HasSubset.Subset N.E (Set.range f)
    hf : Set.InjOn f (Set.preimage f N.E)
    ⊢ ∀ (t : Set α), HasSubset.Subset t (Set.preimage f N.E) → Iff (Exists fun I₀  …
  -/
  refine fun I hI ↦ ⟨fun ⟨I₀, ⟨hI₀, _⟩, hII₀⟩ ↦ ?_, fun h ↦ ⟨_, ⟨h, hf.mono hI⟩, rfl⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    N : Matroid β
    f : α → β
    h_range : HasSubset.Subset N.E (Set.range f)
    hf : Set.InjOn f (Set.preimage f N.E)
    I : Set α
    hI : HasSubset.Subset I (Set.preimage f N.E)
    x✝ : Exists fun I₀ => And (And (N.Indep (Set.image f I₀)) (Set.InjOn f I₀)) (E …
    I₀ : Set α
    hI₀ : N.Indep (Set.image f I₀)
    right✝ : Set.InjOn f I₀
    hII₀ : Eq (Set.image f I) (Set.image f I₀)
    ⊢ N.Indep (Set.image f I)
  -/
  suffices h : I₀ ⊆ f ⁻¹' N.E by rw [InjOn.image_eq_image_iff hf hI h] at hII₀; rwa [hII₀]
  /-
    α : Type u_1
    β : Type u_2
    N : Matroid β
    f : α → β
    h_range : HasSubset.Subset N.E (Set.range f)
    hf : Set.InjOn f (Set.preimage f N.E)
    I : Set α
    hI : HasSubset.Subset I (Set.preimage f N.E)
    x✝ : Exists fun I₀ => And (And (N.Indep (Set.image f I₀)) (Set.InjOn f I₀)) (E …
    I₀ : Set α
    hI₀ : N.Indep (Set.image f I₀)
    right✝ : Set.InjOn f I₀
    hII₀ : Eq (Set.image f I) (Set.image f I₀)
    ⊢ HasSubset.Subset I₀ (Set.preimage f N.E)
  -/
  exact (subset_preimage_image f I₀).trans <| preimage_mono (f := f) hI₀.subset_ground
  /-
    🎉 no goals
  -/


lemma comap_map {f : α → β} (hf : f.Injective) : (M.map f hf.injOn).comap f = M := by
  simp [ext_iff_indep, preimage_image_eq _ hf, and_iff_left hf.injOn,
    image_eq_image hf]


instance [M.Nonempty] {f : α → β} (hf) : (M.map f hf).Nonempty :=
      /-
        α : Type u_1
        β : Type u_2
        f✝ : α → β
        E I : Set α
        M : Matroid α
        N : Matroid β
        inst✝ : M.Nonempty
        f : α → β
        hf : Set.InjOn f M.E
        ⊢ (M.map f hf).E.Nonempty
      -/
  ⟨by simp [M.ground_nonempty]⟩
      /-
        🎉 no goals
      -/


instance [M.Finite] {f : α → β} (hf) : (M.map f hf).Finite :=
  ⟨M.ground_finite.image f⟩


instance [M.Finitary] {f : α → β} (hf) : (M.map f hf).Finitary := by
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N : Matroid β
    inst✝ : M.Finitary
    f : α → β
    hf : Set.InjOn f M.E
    ⊢ (M.map f hf).Finitary
  -/
  refine ⟨fun I hI ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N : Matroid β
    inst✝ : M.Finitary
    f : α → β
    hf : Set.InjOn f M.E
    I : Set β
    hI : ∀ (J : Set β), HasSubset.Subset J I → J.Finite → (M.map f hf).Indep J
    ⊢ (M.map f hf).Indep I
  -/
  simp only [map_indep_iff]
  have h' : I ⊆ f '' M.E := by
    intro e he
    obtain ⟨I₀, hI₀, h_eq⟩ := hI {e} (by simpa) (by simp)
    exact image_subset f hI₀.subset_ground <| h_eq.subset rfl
  /-
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I✝ : Set α
    M : Matroid α
    N : Matroid β
    inst✝ : M.Finitary
    f : α → β
    hf : Set.InjOn f M.E
    I : Set β
    hI : ∀ (J : Set β), HasSubset.Subset J I → J.Finite → (M.map f hf).Indep J
    h' : HasSubset.Subset I (Set.image f M.E)
    ⊢ Exists fun I₀ => And (M.Indep I₀) (Eq I (Set.image f I₀))
  -/
  obtain ⟨I₀, hI₀E, rfl⟩ := subset_image_iff.1 h'
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N : Matroid β
    inst✝ : M.Finitary
    f : α → β
    hf : Set.InjOn f M.E
    I₀ : Set α
    hI₀E : HasSubset.Subset I₀ M.E
    hI : ∀ (J : Set β), HasSubset.Subset J (Set.image f I₀) → J.Finite → (M.map f  …
    h' : HasSubset.Subset (Set.image f I₀) (Set.image f M.E)
    ⊢ Exists fun I₀_1 => And (M.Indep I₀_1) (Eq (Set.image f I₀) (Set.image f I₀_1))
  -/
  refine ⟨I₀, indep_of_forall_finite_subset_indep _ fun J₀ hJ₀I₀ hJ₀ ↦ ?_, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N : Matroid β
    inst✝ : M.Finitary
    f : α → β
    hf : Set.InjOn f M.E
    I₀ : Set α
    hI₀E : HasSubset.Subset I₀ M.E
    hI : ∀ (J : Set β), HasSubset.Subset J (Set.image f I₀) → J.Finite → (M.map f  …
    h' : HasSubset.Subset (Set.image f I₀) (Set.image f M.E)
    J₀ : Set α
    hJ₀I₀ : HasSubset.Subset J₀ I₀
    hJ₀ : J₀.Finite
    ⊢ M.Indep J₀
  -/
  specialize hI (f '' J₀) (image_subset f hJ₀I₀) (hJ₀.image _)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f✝ : α → β
    E I : Set α
    M : Matroid α
    N : Matroid β
    inst✝ : M.Finitary
    f : α → β
    hf : Set.InjOn f M.E
    I₀ : Set α
    hI₀E : HasSubset.Subset I₀ M.E
    h' : HasSubset.Subset (Set.image f I₀) (Set.image f M.E)
    J₀ : Set α
    hJ₀I₀ : HasSubset.Subset J₀ I₀
    hJ₀ : J₀.Finite
    hI : (M.map f hf).Indep (Set.image f J₀)
    ⊢ M.Indep J₀
  -/
  rwa [map_image_indep_iff (hJ₀I₀.trans hI₀E)] at hI
  /-
    🎉 no goals
  -/


instance [M.FiniteRk] {f : α → β} (hf) : (M.map f hf).FiniteRk :=
  let ⟨_, hB⟩ := M.exists_base
  (hB.map hf).finiteRk_of_finite (hB.finite.image _)


instance [M.RkPos] {f : α → β} (hf) : (M.map f hf).RkPos :=
  let ⟨_, hB⟩ := M.exists_base
  (hB.map hf).rkPos_of_nonempty (hB.nonempty.image _)


/-- Map `M : Matroid α` to a `Matroid β` with ground set `E` using an equivalence `M.E ≃ E`.
Defined using `Matroid.ofExistsMatroid` for better defeq. -/
def mapSetEquiv (M : Matroid α) {E : Set β} (e : M.E ≃ E) : Matroid β :=
  Matroid.ofExistsMatroid E (fun I ↦ (M.Indep ↑(e.symm '' (E ↓∩ I)) ∧ I ⊆ E))
  ⟨M.mapSetEmbedding (e.toEmbedding.trans <| Function.Embedding.subtype _), by
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      E✝ I : Set α
      M✝ : Matroid α
      N : Matroid β
      M : Matroid α
      E : Set β
      e : Equiv ↑M.E ↑E
      ⊢ And (Eq E (M.mapSetEmbedding (e.toEmbedding.trans (Function.Embedding.subtyp …
    -/
    have hrw : ∀ I : Set β, Subtype.val ∘ ⇑e ⁻¹' I = ⇑e.symm '' E ↓∩ I := fun I ↦ by ext; simp
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      E✝ I : Set α
      M✝ : Matroid α
      N : Matroid β
      M : Matroid α
      E : Set β
      e : Equiv ↑M.E ↑E
      hrw : ∀ (I : Set β), Eq (Set.preimage (Function.comp Subtype.val ⇑e) I) (Set.i …
      ⊢ And (Eq E (M.mapSetEmbedding (e.toEmbedding.trans (Function.Embedding.subtyp …
    -/
    simp [Equiv.toEmbedding, Embedding.subtype, Embedding.trans, hrw]⟩
    /-
      🎉 no goals
    -/


@[simp] lemma mapSetEquiv_indep_iff (M : Matroid α) {E : Set β} (e : M.E ≃ E) {I : Set β} :
    (M.mapSetEquiv e).Indep I ↔ M.Indep ↑(e.symm '' (E ↓∩ I)) ∧ I ⊆ E := Iff.rfl


@[simp] lemma mapSetEquiv.ground (M : Matroid α) {E : Set β} (e : M.E ≃ E) :
    (M.mapSetEquiv e).E = E := rfl


/-- Map `M : Matroid α` across an embedding defined on all of `α` -/
def mapEmbedding (M : Matroid α) (f : α ↪ β) : Matroid β := M.map f f.injective.injOn


@[simp] lemma mapEmbedding_ground_eq (M : Matroid α) (f : α ↪ β) :
    (M.mapEmbedding f).E = f '' M.E := rfl


@[simp] lemma mapEmbedding_indep_iff {f : α ↪ β} {I : Set β} :
    (M.mapEmbedding f).Indep I ↔ M.Indep (f ⁻¹' I) ∧ I ⊆ range f := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I : Set β
    ⊢ Iff ((M.mapEmbedding f).Indep I) (And (M.Indep (Set.preimage (⇑f) I)) (HasSu …
  -/
  rw [mapEmbedding, map_indep_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I : Set β
    ⊢ Iff (Exists fun I₀ => And (M.Indep I₀) (Eq I (Set.image (⇑f) I₀))) (And (M.I …
  -/
  refine ⟨?_, fun ⟨h,h'⟩ ↦ ⟨f ⁻¹' I, h, by rwa [eq_comm, image_preimage_eq_iff]⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I : Set β
    ⊢ (Exists fun I₀ => And (M.Indep I₀) (Eq I (Set.image (⇑f) I₀))) → And (M.Inde …
  -/
  rintro ⟨I, hI, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I : Set α
    hI : M.Indep I
    ⊢ And (M.Indep (Set.preimage (⇑f) (Set.image (⇑f) I))) (HasSubset.Subset (Set. …
  -/
  rw [preimage_image_eq _ f.injective]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I : Set α
    hI : M.Indep I
    ⊢ And (M.Indep I) (HasSubset.Subset (Set.image (⇑f) I) (Set.range ⇑f))
  -/
  exact ⟨hI, image_subset_range _ _⟩
  /-
    🎉 no goals
  -/


lemma Indep.mapEmbedding (hI : M.Indep I) (f : α ↪ β) : (M.mapEmbedding f).Indep (f '' I) := by
  /-
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    hI : M.Indep I
    f : Function.Embedding α β
    ⊢ (M.mapEmbedding f).Indep (Set.image (⇑f) I)
  -/
  simpa [preimage_image_eq I f.injective]
  /-
    🎉 no goals
  -/


lemma Base.mapEmbedding {B : Set α} (hB : M.Base B) (f : α ↪ β) :
    (M.mapEmbedding f).Base (f '' B) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    B : Set α
    hB : M.Base B
    f : Function.Embedding α β
    ⊢ (M.mapEmbedding f).Base (Set.image (⇑f) B)
  -/
  rw [Matroid.mapEmbedding, map_base_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    B : Set α
    hB : M.Base B
    f : Function.Embedding α β
    ⊢ Exists fun B₀ => And (M.Base B₀) (Eq (Set.image (⇑f) B) (Set.image (⇑f) B₀))
  -/
  exact ⟨B, hB, rfl⟩
  /-
    🎉 no goals
  -/


lemma Basis.mapEmbedding {X : Set α} (hIX : M.Basis I X) (f : α ↪ β) :
    (M.mapEmbedding f).Basis (f '' I) (f '' X) := by
  /-
    α : Type u_1
    β : Type u_2
    I : Set α
    M : Matroid α
    X : Set α
    hIX : M.Basis I X
    f : Function.Embedding α β
    ⊢ (M.mapEmbedding f).Basis (Set.image (⇑f) I) (Set.image (⇑f) X)
  -/
  apply hIX.map
  /-
    🎉 no goals
  -/


@[simp] lemma mapEmbedding_base_iff {f : α ↪ β} {B : Set β} :
    (M.mapEmbedding f).Base B ↔ M.Base (f ⁻¹' B) ∧ B ⊆ range f := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    B : Set β
    ⊢ Iff ((M.mapEmbedding f).Base B) (And (M.Base (Set.preimage (⇑f) B)) (HasSubs …
  -/
  rw [mapEmbedding, map_base_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    B : Set β
    ⊢ Iff (Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image (⇑f) B₀))) (And (M.Ba …
  -/
  refine ⟨?_, fun ⟨h,h'⟩ ↦ ⟨f ⁻¹' B, h, by rwa [eq_comm, image_preimage_eq_iff]⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    B : Set β
    ⊢ (Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image (⇑f) B₀))) → And (M.Base  …
  -/
  rintro ⟨B, hB, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    B : Set α
    hB : M.Base B
    ⊢ And (M.Base (Set.preimage (⇑f) (Set.image (⇑f) B))) (HasSubset.Subset (Set.i …
  -/
  rw [preimage_image_eq _ f.injective]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    B : Set α
    hB : M.Base B
    ⊢ And (M.Base B) (HasSubset.Subset (Set.image (⇑f) B) (Set.range ⇑f))
  -/
  exact ⟨hB, image_subset_range _ _⟩
  /-
    🎉 no goals
  -/


@[simp] lemma mapEmbedding_basis_iff {f : α ↪ β} {I X : Set β} :
    (M.mapEmbedding f).Basis I X ↔ M.Basis (f ⁻¹' I) (f ⁻¹' X) ∧ I ⊆ X ∧ X ⊆ range f := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I X : Set β
    ⊢ Iff ((M.mapEmbedding f).Basis I X) (And (M.Basis (Set.preimage (⇑f) I) (Set. …
  -/
  rw [mapEmbedding, map_basis_iff']
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I X : Set β
    ⊢ Iff (Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.i …
  -/
  refine ⟨?_, fun ⟨hb, hIX, hX⟩ ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : Function.Embedding α β
      I X : Set β
      ⊢ (Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.image …
    -/
  · rintro ⟨I, X, hIX, rfl, rfl⟩
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      M : Matroid α
      f : Function.Embedding α β
      I X : Set α
      hIX : M.Basis I X
      ⊢ And (M.Basis (Set.preimage (⇑f) (Set.image (⇑f) I)) (Set.preimage (⇑f) (Set. …
    -/
    simp [preimage_image_eq _ f.injective, image_subset f hIX.subset, hIX]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I X : Set β
    x✝ : And (M.Basis (Set.preimage (⇑f) I) (Set.preimage (⇑f) X)) (And (HasSubset …
    hb : M.Basis (Set.preimage (⇑f) I) (Set.preimage (⇑f) X)
    hIX : HasSubset.Subset I X
    hX : HasSubset.Subset X (Set.range ⇑f)
    ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.image  …
  -/
  obtain ⟨X, rfl⟩ := subset_range_iff_exists_image_eq.1 hX
  /-
    case refine_2.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    I : Set β
    X : Set α
    x✝ : And (M.Basis (Set.preimage (⇑f) I) (Set.preimage (⇑f) (Set.image (⇑f) X)) …
    hb : M.Basis (Set.preimage (⇑f) I) (Set.preimage (⇑f) (Set.image (⇑f) X))
    hIX : HasSubset.Subset I (Set.image (⇑f) X)
    hX : HasSubset.Subset (Set.image (⇑f) X) (Set.range ⇑f)
    ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.image  …
  -/
  obtain ⟨I, -, rfl⟩ := subset_image_iff.1 hIX
  /-
    case refine_2.intro.intro.intro
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Function.Embedding α β
    X : Set α
    hX : HasSubset.Subset (Set.image (⇑f) X) (Set.range ⇑f)
    I : Set α
    x✝ : And (M.Basis (Set.preimage (⇑f) (Set.image (⇑f) I)) (Set.preimage (⇑f) (S …
    hb : M.Basis (Set.preimage (⇑f) (Set.image (⇑f) I)) (Set.preimage (⇑f) (Set.im …
    hIX : HasSubset.Subset (Set.image (⇑f) I) (Set.image (⇑f) X)
    ⊢ Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq (Set.image (⇑ …
  -/
  exact ⟨I, X, by simpa [preimage_image_eq _ f.injective] using hb⟩
  /-
    🎉 no goals
  -/


instance [M.Nonempty] {f : α ↪ β} : (M.mapEmbedding f).Nonempty :=
  inferInstanceAs (M.map f f.injective.injOn).Nonempty


instance [M.Finite] {f : α ↪ β} : (M.mapEmbedding f).Finite :=
  inferInstanceAs (M.map f f.injective.injOn).Finite


instance [M.Finitary] {f : α ↪ β} : (M.mapEmbedding f).Finitary :=
  inferInstanceAs (M.map f f.injective.injOn).Finitary


instance [M.FiniteRk] {f : α ↪ β} : (M.mapEmbedding f).FiniteRk :=
  inferInstanceAs (M.map f f.injective.injOn).FiniteRk


instance [M.RkPos] {f : α ↪ β} : (M.mapEmbedding f).RkPos :=
  inferInstanceAs (M.map f f.injective.injOn).RkPos


/-- Map `M : Matroid α` across an equivalence `α ≃ β` -/
def mapEquiv (M : Matroid α) (f : α ≃ β) : Matroid β := M.mapEmbedding f.toEmbedding


@[simp] lemma mapEquiv_ground_eq (M : Matroid α) (f : α ≃ β) :
    (M.mapEquiv f).E = f '' M.E := rfl


lemma mapEquiv_eq_map (f : α ≃ β) : M.mapEquiv f = M.map f f.injective.injOn := rfl


@[simp] lemma mapEquiv_indep_iff {I : Set β} : (M.mapEquiv f).Indep I ↔ M.Indep (f.symm '' I) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Equiv α β
    I : Set β
    ⊢ Iff ((M.mapEquiv f).Indep I) (M.Indep (Set.image (⇑f.symm) I))
  -/
  rw [mapEquiv_eq_map, map_indep_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Equiv α β
    I : Set β
    ⊢ Iff (Exists fun I₀ => And (M.Indep I₀) (Eq I (Set.image (⇑f) I₀))) (M.Indep  …
  -/
  exact ⟨by rintro ⟨I, hI, rfl⟩; simpa, fun h ↦ ⟨_, h, by simp⟩⟩
  /-
    🎉 no goals
  -/


@[simp] lemma mapEquiv_dep_iff {D : Set β} : (M.mapEquiv f).Dep D ↔ M.Dep (f.symm '' D) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Equiv α β
    D : Set β
    ⊢ Iff ((M.mapEquiv f).Dep D) (M.Dep (Set.image (⇑f.symm) D))
  -/
  rw [mapEquiv_eq_map, map_dep_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Equiv α β
    D : Set β
    ⊢ Iff (Exists fun D₀ => And (M.Dep D₀) (Eq D (Set.image (⇑f) D₀))) (M.Dep (Set …
  -/
  exact ⟨by rintro ⟨I, hI, rfl⟩; simpa, fun h ↦ ⟨_, h, by simp⟩⟩
  /-
    🎉 no goals
  -/


@[simp] lemma mapEquiv_base_iff {B : Set β} : (M.mapEquiv f).Base B ↔ M.Base (f.symm '' B) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Equiv α β
    B : Set β
    ⊢ Iff ((M.mapEquiv f).Base B) (M.Base (Set.image (⇑f.symm) B))
  -/
  rw [mapEquiv_eq_map, map_base_iff]
  /-
    α : Type u_1
    β : Type u_2
    M : Matroid α
    f : Equiv α β
    B : Set β
    ⊢ Iff (Exists fun B₀ => And (M.Base B₀) (Eq B (Set.image (⇑f) B₀))) (M.Base (S …
  -/
  exact ⟨by rintro ⟨I, hI, rfl⟩; simpa, fun h ↦ ⟨_, h, by simp⟩⟩
  /-
    🎉 no goals
  -/


@[simp] lemma mapEquiv_basis_iff {α β : Type*} {M : Matroid α} (f : α ≃ β) {I X : Set β} :
    (M.mapEquiv f).Basis I X ↔ M.Basis (f.symm '' I) (f.symm '' X) := by
  /-
    α : Type u_3
    β : Type u_4
    M : Matroid α
    f : Equiv α β
    I X : Set β
    ⊢ Iff ((M.mapEquiv f).Basis I X) (M.Basis (Set.image (⇑f.symm) I) (Set.image ( …
  -/
  rw [mapEquiv_eq_map, map_basis_iff']
  /-
    α : Type u_3
    β : Type u_4
    M : Matroid α
    f : Equiv α β
    I X : Set β
    ⊢ Iff (Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.i …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ⟨_, _, h, by simp, by simp⟩⟩
  /-
    α : Type u_3
    β : Type u_4
    M : Matroid α
    f : Equiv α β
    I X : Set β
    h : Exists fun I₀ => Exists fun X₀ => And (M.Basis I₀ X₀) (And (Eq I (Set.imag …
    ⊢ M.Basis (Set.image (⇑f.symm) I) (Set.image (⇑f.symm) X)
  -/
  obtain ⟨I, X, hIX, rfl, rfl⟩ := h
  /-
    case intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    M : Matroid α
    f : Equiv α β
    I X : Set α
    hIX : M.Basis I X
    ⊢ M.Basis (Set.image (⇑f.symm) (Set.image (⇑f) I)) (Set.image (⇑f.symm) (Set.i …
  -/
  simpa
  /-
    🎉 no goals
  -/


instance [M.Nonempty] {f : α ≃ β} : (M.mapEquiv f).Nonempty :=
  inferInstanceAs (M.map f f.injective.injOn).Nonempty


instance [M.Finite] {f : α ≃ β} : (M.mapEquiv f).Finite :=
  inferInstanceAs (M.map f f.injective.injOn).Finite


instance [M.Finitary] {f : α ≃ β} : (M.mapEquiv f).Finitary :=
  inferInstanceAs (M.map f f.injective.injOn).Finitary


instance [M.FiniteRk] {f : α ≃ β} : (M.mapEquiv f).FiniteRk :=
  inferInstanceAs (M.map f f.injective.injOn).FiniteRk


instance [M.RkPos] {f : α ≃ β} : (M.mapEquiv f).RkPos :=
  inferInstanceAs (M.map f f.injective.injOn).RkPos


/-- Given `M : Matroid α` and `X : Set α`, the restriction of `M` to `X`,
viewed as a matroid on type `X` with ground set `univ`.
Always isomorphic to `M ↾ X`. If `X = M.E`, then isomorphic to `M`. -/
def restrictSubtype (M : Matroid α) (X : Set α) : Matroid X := (M ↾ X).comap (↑)


@[simp] lemma restrictSubtype_ground : (M.restrictSubtype X).E = univ := by
  /-
    α : Type u_1
    X : Set α
    M : Matroid α
    ⊢ Eq (M.restrictSubtype X).E Set.univ
  -/
  simp [restrictSubtype]
  /-
    🎉 no goals
  -/


@[simp] lemma restrictSubtype_indep_iff {I : Set X} :
    (M.restrictSubtype X).Indep I ↔ M.Indep ((↑) '' I) := by
  /-
    α : Type u_1
    X : Set α
    M : Matroid α
    I : Set ↑X
    ⊢ Iff ((M.restrictSubtype X).Indep I) (M.Indep ↑I)
  -/
  simp [restrictSubtype, Subtype.val_injective.injOn]
  /-
    🎉 no goals
  -/


lemma restrictSubtype_indep_iff_of_subset (hIX : I ⊆ X) :
    (M.restrictSubtype X).Indep (X ↓∩ I) ↔ M.Indep I := by
  /-
    α : Type u_1
    X I : Set α
    M : Matroid α
    hIX : HasSubset.Subset I X
    ⊢ Iff ((M.restrictSubtype X).Indep (Set.preimage Subtype.val I)) (M.Indep I)
  -/
  rw [restrictSubtype_indep_iff, image_preimage_eq_iff.2]; simpa
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma restrictSubtype_inter_indep_iff :
    (M.restrictSubtype X).Indep (X ↓∩ I) ↔ M.Indep (X ∩ I) := by
  /-
    α : Type u_1
    X I : Set α
    M : Matroid α
    ⊢ Iff ((M.restrictSubtype X).Indep (Set.preimage Subtype.val I)) (M.Indep (Int …
  -/
  simp [restrictSubtype, Subtype.val_injective.injOn]
  /-
    🎉 no goals
  -/


lemma restrictSubtype_basis_iff {Y : Set α} {I X : Set Y} :
    (M.restrictSubtype Y).Basis I X ↔ M.Basis' I X := by
  rw [restrictSubtype, comap_basis_iff, and_iff_right Subtype.val_injective.injOn,
    and_iff_left_of_imp, basis_restrict_iff', basis'_iff_basis_inter_ground]
    /-
      α : Type u_1
      M : Matroid α
      Y : Set α
      I X : Set ↑Y
      ⊢ Iff (And (M.Basis (↑I) (Inter.inter (↑X) M.E)) (HasSubset.Subset (↑X) Y)) (M …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    M : Matroid α
    Y : Set α
    I X : Set ↑Y
    ⊢ (M.restrict Y).Basis ↑I ↑X → HasSubset.Subset I X
  -/
  exact fun h ↦ (image_subset_image_iff Subtype.val_injective).1 h.subset
  /-
    🎉 no goals
  -/


lemma restrictSubtype_base_iff {B : Set X} : (M.restrictSubtype X).Base B ↔ M.Basis' B X := by
  /-
    α : Type u_1
    X : Set α
    M : Matroid α
    B : Set ↑X
    ⊢ Iff ((M.restrictSubtype X).Base B) (M.Basis' (Set.image Subtype.val B) X)
  -/
  rw [restrictSubtype, comap_base_iff]
  /-
    α : Type u_1
    X : Set α
    M : Matroid α
    B : Set ↑X
    ⊢ Iff (And ((M.restrict X).Basis ↑B ↑(Set.preimage Subtype.val (M.restrict X). …
  -/
  simp [Subtype.val_injective.injOn, Subset.rfl, basis_restrict_iff', basis'_iff_basis_inter_ground]
  /-
    🎉 no goals
  -/


@[simp] lemma restrictSubtype_ground_base_iff {B : Set M.E} :
    (M.restrictSubtype M.E).Base B ↔ M.Base B := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set ↑M.E
    ⊢ Iff ((M.restrictSubtype M.E).Base B) (M.Base (Set.image Subtype.val B))
  -/
  rw [restrictSubtype_base_iff, basis'_iff_basis, basis_ground_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma restrictSubtype_ground_basis_iff {I X : Set M.E} :
    (M.restrictSubtype M.E).Basis I X ↔ M.Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set ↑M.E
    ⊢ Iff ((M.restrictSubtype M.E).Basis I X) (M.Basis (Set.image Subtype.val I) ( …
  -/
  rw [restrictSubtype_basis_iff, basis'_iff_basis]
  /-
    🎉 no goals
  -/


lemma eq_of_restrictSubtype_eq {N : Matroid α} (hM : M.E = E) (hN : N.E = E)
    (h : M.restrictSubtype E = N.restrictSubtype E) : M = N := by
  /-
    α : Type u_1
    E : Set α
    M N : Matroid α
    hM : Eq M.E E
    hN : Eq N.E E
    h : Eq (M.restrictSubtype E) (N.restrictSubtype E)
    ⊢ Eq M N
  -/
  subst hM
  /-
    α : Type u_1
    M N : Matroid α
    hN : Eq N.E M.E
    h : Eq (M.restrictSubtype M.E) (N.restrictSubtype M.E)
    ⊢ Eq M N
  -/
  refine ext_indep (by rw [hN]) (fun I hI ↦ ?_)
  /-
    α : Type u_1
    M N : Matroid α
    hN : Eq N.E M.E
    h : Eq (M.restrictSubtype M.E) (N.restrictSubtype M.E)
    I : Set α
    hI : HasSubset.Subset I M.E
    ⊢ Iff (M.Indep I) (N.Indep I)
  -/
  rwa [← restrictSubtype_indep_iff_of_subset hI, h, restrictSubtype_indep_iff_of_subset]
  /-
    🎉 no goals
  -/


@[simp] lemma restrictSubtype_dual : (M.restrictSubtype M.E)✶ = M✶.restrictSubtype M.E := by
  rw [restrictSubtype, ← comapOn_preimage_eq, comapOn_dual_eq_of_bijOn, restrict_ground_eq_self,
    ← dual_ground, comapOn_preimage_eq, restrictSubtype, restrict_ground_eq_self]
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Set.BijOn Subtype.val (Set.preimage Subtype.val (M.restrict M.E).E) (M.restr …
  -/
  exact ⟨by simp [MapsTo], Subtype.val_injective.injOn, by simp [SurjOn, Subset.rfl]⟩
  /-
    🎉 no goals
  -/


lemma restrictSubtype_dual' (hM : M.E = E) : (M.restrictSubtype E)✶ = M✶.restrictSubtype E := by
  /-
    α : Type u_1
    E : Set α
    M : Matroid α
    hM : Eq M.E E
    ⊢ Eq (M.restrictSubtype E).dual (M.dual.restrictSubtype E)
  -/
  rw [← hM, restrictSubtype_dual]
  /-
    🎉 no goals
  -/


/-- `M.restrictSubtype X` is isomorphic to `M ↾ X`. -/
@[simp] lemma map_val_restrictSubtype_eq (M : Matroid α) (X : Set α) :
    (M.restrictSubtype X).map (↑) Subtype.val_injective.injOn = M ↾ X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    ⊢ Eq ((M.restrictSubtype X).map Subtype.val ⋯) (M.restrict X)
  -/
  simp [restrictSubtype, map_comap, Subset.rfl]
  /-
    🎉 no goals
  -/


/-- `M.restrictSubtype M.E` is isomorphic to `M`. -/
lemma map_val_restrictSubtype_ground_eq (M : Matroid α) :
    (M.restrictSubtype M.E).map (↑) Subtype.val_injective.injOn = M := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq ((M.restrictSubtype M.E).map Subtype.val ⋯) M
  -/
  simp
  /-
    🎉 no goals
  -/


instance [M.Finitary] {X : Set α} : (M.restrictSubtype X).Finitary := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M✝ : Matroid α
    N : Matroid β
    E X✝ I : Set α
    M : Matroid α
    inst✝ : M.Finitary
    X : Set α
    ⊢ (M.restrictSubtype X).Finitary
  -/
  rw [restrictSubtype]; infer_instance
                        /-
                          🎉 no goals
                        -/


instance [M.FiniteRk] {X : Set α} : (M.restrictSubtype X).FiniteRk := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M✝ : Matroid α
    N : Matroid β
    E X✝ I : Set α
    M : Matroid α
    inst✝ : M.FiniteRk
    X : Set α
    ⊢ (M.restrictSubtype X).FiniteRk
  -/
  rw [restrictSubtype]; infer_instance
                        /-
                          🎉 no goals
                        -/


instance [M.Finite] : (M.restrictSubtype M.E).Finite :=
  have := M.ground_finite.to_subtype
  ⟨Finite.ground_finite⟩


instance [M.Nonempty] : (M.restrictSubtype M.E).Nonempty :=
  have := M.ground_nonempty.coe_sort
      /-
        α : Type u_1
        β : Type u_2
        f : α → β
        E✝ I✝ : Set α
        M✝ : Matroid α
        N : Matroid β
        E X I : Set α
        M : Matroid α
        inst✝ : M.Nonempty
        this : Nonempty ↑M.E
        ⊢ (M.restrictSubtype M.E).E.Nonempty
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance [M.RkPos] : (M.restrictSubtype M.E).RkPos := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M✝ : Matroid α
    N : Matroid β
    E X I : Set α
    M : Matroid α
    inst✝ : M.RkPos
    ⊢ (M.restrictSubtype M.E).RkPos
  -/
  obtain ⟨B, hB⟩ := (M.restrictSubtype M.E).exists_base
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M✝ : Matroid α
    N : Matroid β
    E X I : Set α
    M : Matroid α
    inst✝ : M.RkPos
    B : Set ↑M.E
    hB : (M.restrictSubtype M.E).Base B
    ⊢ (M.restrictSubtype M.E).RkPos
  -/
  have hB' : M.Base ↑B := by simpa using hB.map Subtype.val_injective.injOn
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    E✝ I✝ : Set α
    M✝ : Matroid α
    N : Matroid β
    E X I : Set α
    M : Matroid α
    inst✝ : M.RkPos
    B : Set ↑M.E
    hB : (M.restrictSubtype M.E).Base B
    hB' : M.Base (Set.image Subtype.val B)
    ⊢ (M.restrictSubtype M.E).RkPos
  -/
  exact hB.rkPos_of_nonempty <| by simpa using hB'.nonempty
  /-
    🎉 no goals
  -/


