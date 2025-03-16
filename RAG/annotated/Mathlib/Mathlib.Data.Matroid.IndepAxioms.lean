/-- A matroid as defined by the independence axioms. This is the same thing as a `Matroid`,
  and so does not need its own API; it exists to make it easier to construct a matroid from its
  independent sets. The constructed `IndepMatroid` can then be converted into a matroid
  with `IndepMatroid.matroid`. -/
structure IndepMatroid (α : Type*) where
  /-- The ground set -/
  (E : Set α)
  /-- The independence predicate -/
  (Indep : Set α → Prop)
  (indep_empty : Indep ∅)
  (indep_subset : ∀ ⦃I J⦄, Indep J → I ⊆ J → Indep I)
  (indep_aug : ∀ ⦃I B⦄, Indep I → ¬ Maximal Indep I →
    Maximal Indep B → ∃ x ∈ B \ I, Indep (insert x I))
  (indep_maximal : ∀ X, X ⊆ E → ExistsMaximalSubsetProperty Indep X)
  (subset_ground : ∀ I, Indep I → I ⊆ E)


/-- An `M : IndepMatroid α` gives a `Matroid α` whose bases are the maximal `M`-independent sets. -/
@[simps] protected def matroid (M : IndepMatroid α) : Matroid α where
  E := M.E
  Base := Maximal M.Indep
  Indep := M.Indep
  indep_iff' := by
    /-
      α : Type u_1
      M : IndepMatroid α
      ⊢ ∀ ⦃I : Set α⦄, Iff (M.Indep I) (Exists fun B => And (Maximal M.Indep B) (Has …
    -/
    refine fun I ↦ ⟨fun h ↦ ?_, fun ⟨B, ⟨h, _⟩, hIB'⟩ ↦ M.indep_subset h hIB'⟩
    /-
      α : Type u_1
      M : IndepMatroid α
      I : Set α
      h : M.Indep I
      ⊢ Exists fun B => And (Maximal M.Indep B) (HasSubset.Subset I B)
    -/
    obtain ⟨J, hIJ, hmax⟩ := M.indep_maximal M.E rfl.subset I h (M.subset_ground I h)
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      I : Set α
      h : M.Indep I
      J : Set α
      hIJ : HasSubset.Subset I J
      hmax : Maximal (fun K => And (M.Indep K) (HasSubset.Subset K M.E)) J
      ⊢ Exists fun B => And (Maximal M.Indep B) (HasSubset.Subset I B)
    -/
    rw [maximal_and_iff_right_of_imp M.subset_ground] at hmax
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      I : Set α
      h : M.Indep I
      J : Set α
      hIJ : HasSubset.Subset I J
      hmax : And (Maximal M.Indep J) (HasSubset.Subset J M.E)
      ⊢ Exists fun B => And (Maximal M.Indep B) (HasSubset.Subset I B)
    -/
    exact ⟨J, hmax.1, hIJ⟩
    /-
      🎉 no goals
    -/
  exists_base := by
    /-
      α : Type u_1
      M : IndepMatroid α
      ⊢ Exists fun B => Maximal M.Indep B
    -/
    obtain ⟨B, -, hB⟩ := M.indep_maximal M.E rfl.subset ∅ M.indep_empty <| empty_subset _
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      B : Set α
      hB : Maximal (fun K => And (M.Indep K) (HasSubset.Subset K M.E)) B
      ⊢ Exists fun B => Maximal M.Indep B
    -/
    rw [maximal_and_iff_right_of_imp M.subset_ground] at hB
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      B : Set α
      hB : And (Maximal M.Indep B) (HasSubset.Subset B M.E)
      ⊢ Exists fun B => Maximal M.Indep B
    -/
    exact ⟨B, hB.1⟩
    /-
      🎉 no goals
    -/
  base_exchange B B' hB hB' e he := by
    have hnotmax : ¬ Maximal M.Indep (B \ {e}) :=
      fun h ↦ h.not_prop_of_ssuperset (diff_singleton_sSubset.2 he.1) hB.prop
    /-
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      e : α
      he : Membership.mem (SDiff.sdiff B B') e
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton e)))
      ⊢ Exists fun b => And (Membership.mem (SDiff.sdiff B' B) b) (Maximal M.Indep ( …
    -/
    obtain ⟨f, hf, hfB⟩ := M.indep_aug (M.indep_subset hB.prop diff_subset) hnotmax hB'
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      e : α
      he : Membership.mem (SDiff.sdiff B B') e
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton e)))
      f : α
      hf : Membership.mem (SDiff.sdiff B' (SDiff.sdiff B (Singleton.singleton e))) f
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
      ⊢ Exists fun b => And (Membership.mem (SDiff.sdiff B' B) b) (Maximal M.Indep ( …
    -/
    replace hf := show f ∈ B' \ B by simpa [show f ≠ e by rintro rfl; exact he.2 hf.1] using hf
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      e : α
      he : Membership.mem (SDiff.sdiff B B') e
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton e)))
      f : α
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
      hf : Membership.mem (SDiff.sdiff B' B) f
      ⊢ Exists fun b => And (Membership.mem (SDiff.sdiff B' B) b) (Maximal M.Indep ( …
    -/
    refine ⟨f, hf, by_contra fun hnot ↦ ?_⟩
    /-
      case intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      e : α
      he : Membership.mem (SDiff.sdiff B B') e
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton e)))
      f : α
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
      hf : Membership.mem (SDiff.sdiff B' B) f
      hnot : Not (Maximal M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singlet …
      ⊢ False
    -/
    obtain ⟨x, hxB, hind⟩ := M.indep_aug hfB hnot hB
    /-
      case intro.intro.intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      e : α
      he : Membership.mem (SDiff.sdiff B B') e
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton e)))
      f : α
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
      hf : Membership.mem (SDiff.sdiff B' B) f
      hnot : Not (Maximal M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singlet …
      x : α
      hxB : Membership.mem (SDiff.sdiff B (Insert.insert f (SDiff.sdiff B (Singleton …
      hind : M.Indep (Insert.insert x (Insert.insert f (SDiff.sdiff B (Singleton.sin …
      ⊢ False
    -/
    obtain ⟨-, rfl⟩ : _ ∧ x = e := by simpa [hxB.1] using hxB
    /-
      case intro.intro.intro.intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      f : α
      hf : Membership.mem (SDiff.sdiff B' B) f
      x : α
      he : Membership.mem (SDiff.sdiff B B') x
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton x)))
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton x)))
      hnot : Not (Maximal M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singlet …
      hxB : Membership.mem (SDiff.sdiff B (Insert.insert f (SDiff.sdiff B (Singleton …
      hind : M.Indep (Insert.insert x (Insert.insert f (SDiff.sdiff B (Singleton.sin …
      ⊢ False
    -/
    refine hB.not_prop_of_ssuperset ?_ hind
    /-
      case intro.intro.intro.intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      f : α
      hf : Membership.mem (SDiff.sdiff B' B) f
      x : α
      he : Membership.mem (SDiff.sdiff B B') x
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton x)))
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton x)))
      hnot : Not (Maximal M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singlet …
      hxB : Membership.mem (SDiff.sdiff B (Insert.insert f (SDiff.sdiff B (Singleton …
      hind : M.Indep (Insert.insert x (Insert.insert f (SDiff.sdiff B (Singleton.sin …
      ⊢ HasSSubset.SSubset B (Insert.insert x (Insert.insert f (SDiff.sdiff B (Singl …
    -/
    rw [insert_comm, insert_diff_singleton, insert_eq_of_mem he.1]
    /-
      case intro.intro.intro.intro.intro
      α : Type u_1
      M : IndepMatroid α
      B B' : Set α
      hB : Maximal M.Indep B
      hB' : Maximal M.Indep B'
      f : α
      hf : Membership.mem (SDiff.sdiff B' B) f
      x : α
      he : Membership.mem (SDiff.sdiff B B') x
      hnotmax : Not (Maximal M.Indep (SDiff.sdiff B (Singleton.singleton x)))
      hfB : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton x)))
      hnot : Not (Maximal M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singlet …
      hxB : Membership.mem (SDiff.sdiff B (Insert.insert f (SDiff.sdiff B (Singleton …
      hind : M.Indep (Insert.insert x (Insert.insert f (SDiff.sdiff B (Singleton.sin …
      ⊢ HasSSubset.SSubset B (Insert.insert f B)
    -/
    exact ssubset_insert hf.2
    /-
      🎉 no goals
    -/
  maximality := M.indep_maximal
  subset_ground B hB := M.subset_ground B hB.1


@[simp] theorem matroid_indep_iff {M : IndepMatroid α} {I : Set α} :
    M.matroid.Indep I ↔ M.Indep I := Iff.rfl


/-- An independence predicate satisfying the finite matroid axioms determines a matroid,
  provided independence is determined by its behaviour on finite sets.
  This fundamentally needs choice, since it can be used to prove that every vector space
  has a basis. -/
@[simps E] protected def ofFinitary (E : Set α) (Indep : Set α → Prop)
    (indep_empty : Indep ∅)
    (indep_subset : ∀ ⦃I J⦄, Indep J → I ⊆ J → Indep I)
    (indep_aug : ∀ ⦃I J⦄, Indep I → I.Finite → Indep J → J.Finite → I.ncard < J.ncard →
      ∃ e ∈ J, e ∉ I ∧ Indep (insert e I))
    (indep_compact : ∀ I, (∀ J, J ⊆ I → J.Finite → Indep J) → Indep I)
    (subset_ground : ∀ I, Indep I → I ⊆ E) : IndepMatroid α :=
  have htofin : ∀ I e, Indep I → ¬ Indep (insert e I) →
    ∃ I₀, I₀ ⊆ I ∧ I₀.Finite ∧ ¬ Indep (insert e I₀) := by
      /-
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        ⊢ ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Exists fu …
      -/
      by_contra h; push_neg at h
      /-
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        h : Exists fun I => Exists fun e => And (Indep I) (And (Not (Indep (Insert.ins …
        ⊢ False
      -/
      obtain ⟨I, e, -, hIe, h⟩ := h
      /-
        case intro.intro.intro.intro
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        I : Set α
        e : α
        hIe : Not (Indep (Insert.insert e I))
        h : ∀ (I₀ : Set α), HasSubset.Subset I₀ I → I₀.Finite → Indep (Insert.insert e …
        ⊢ False
      -/
      refine hIe <| indep_compact _ fun J hJss hJfin ↦ ?_
      /-
        case intro.intro.intro.intro
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        I : Set α
        e : α
        hIe : Not (Indep (Insert.insert e I))
        h : ∀ (I₀ : Set α), HasSubset.Subset I₀ I → I₀.Finite → Indep (Insert.insert e …
        J : Set α
        hJss : HasSubset.Subset J (Insert.insert e I)
        hJfin : J.Finite
        ⊢ Indep J
      -/
      exact indep_subset (h (J \ {e}) (by rwa [diff_subset_iff]) (hJfin.diff _)) (by simp)
      /-
        🎉 no goals
      -/
  IndepMatroid.mk
  (E := E)
  (Indep := Indep)
  (indep_empty := indep_empty)
  (indep_subset := indep_subset)
  (indep_aug := by
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      ⊢ ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B → Exists  …
    -/
    intro I B hI hImax hBmax
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) (Indep (Insert.inse …
    -/
    obtain ⟨e, heI, hins⟩ := exists_insert_of_not_maximal indep_subset hI hImax
    /-
      case intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) (Indep (Insert.inse …
    -/
    by_cases heB : e ∈ B
      /-
        case pos
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
        I B : Set α
        hI : Indep I
        hImax : Not (Maximal Indep I)
        hBmax : Maximal Indep B
        e : α
        heI : Not (Membership.mem I e)
        hins : Indep (Insert.insert e I)
        heB : Membership.mem B e
        ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) (Indep (Insert.inse …
      -/
    · exact ⟨e, ⟨heB, heI⟩, hins⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) (Indep (Insert.inse …
    -/
    by_contra hcon; push_neg at hcon

    /-
      case neg
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      ⊢ False
    -/
    have heBdep := hBmax.not_prop_of_ssuperset (ssubset_insert heB)

    -- There is a finite subset `B₀` of `B` so that `B₀ + e` is dependent
    /-
      case neg
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      ⊢ False
    -/
    obtain ⟨B₀, hB₀B, hB₀fin, hB₀e⟩ := htofin B e hBmax.1 heBdep
    /-
      case neg.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      ⊢ False
    -/
    have hB₀ := indep_subset hBmax.1 hB₀B

    -- `I` has a finite subset `I₀` that doesn't extend into `B₀`
    have hexI₀ : ∃ I₀, I₀ ⊆ I ∧ I₀.Finite ∧ ∀ x, x ∈ B₀ \ I₀ → ¬Indep (insert x I₀) := by
      have hchoose : ∀ (b : ↑(B₀ \ I)), ∃ Ib, Ib ⊆ I ∧ Ib.Finite ∧ ¬Indep (insert (b : α) Ib) := by
        rintro ⟨b, hb⟩; exact htofin I b hI (hcon b ⟨hB₀B hb.1, hb.2⟩)
      choose! f hf using hchoose
      have := (hB₀fin.diff I).to_subtype
      refine ⟨iUnion f ∪ (B₀ ∩ I),
        union_subset (iUnion_subset (fun i ↦ (hf i).1)) inter_subset_right,
        (finite_iUnion fun i ↦ (hf i).2.1).union (hB₀fin.subset inter_subset_left),
        fun x ⟨hxB₀, hxn⟩ hi ↦ ?_⟩
      have hxI : x ∉ I := fun hxI ↦ hxn <| Or.inr ⟨hxB₀, hxI⟩
      refine (hf ⟨x, ⟨hxB₀, hxI⟩⟩).2.2 (indep_subset hi <| insert_subset_insert ?_)
      apply subset_union_of_subset_left
      apply subset_iUnion

    /-
      case neg.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      hexI₀ : Exists fun I₀ => And (HasSubset.Subset I₀ I) (And I₀.Finite (∀ (x : α) …
      ⊢ False
    -/
    obtain ⟨I₀, hI₀I, hI₀fin, hI₀⟩ := hexI₀

    /-
      case neg.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      ⊢ False
    -/
    set E₀ := insert e (I₀ ∪ B₀)
    /-
      case neg.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      ⊢ False
    -/
    have hE₀fin : E₀.Finite := (hI₀fin.union hB₀fin).insert e

    -- Extend `B₀` to a maximal independent subset of `I₀ ∪ B₀ + e`
    obtain ⟨J, ⟨hB₀J, hJ, hJss⟩, hJmax⟩ := Finite.exists_maximal_wrt (f := id)
      (s := {J | B₀ ⊆ J ∧ Indep J ∧ J ⊆ E₀})
      (hE₀fin.finite_subsets.subset (by simp))
      ⟨B₀, Subset.rfl, hB₀, subset_union_right.trans (subset_insert _ _)⟩

    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      ⊢ False
    -/
    have heI₀ : e ∉ I₀ := not_mem_subset hI₀I heI
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      heI₀ : Not (Membership.mem I₀ e)
      ⊢ False
    -/
    have heI₀i : Indep (insert e I₀) := indep_subset hins (insert_subset_insert hI₀I)

    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      heI₀ : Not (Membership.mem I₀ e)
      heI₀i : Indep (Insert.insert e I₀)
      ⊢ False
    -/
    have heJ : e ∉ J := fun heJ ↦ hB₀e (indep_subset hJ <| insert_subset heJ hB₀J)

    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      heI₀ : Not (Membership.mem I₀ e)
      heI₀i : Indep (Insert.insert e I₀)
      heJ : Not (Membership.mem J e)
      ⊢ False
    -/
    have hJfin := hE₀fin.subset hJss

    -- We have `|I₀ + e| ≤ |J|`, since otherwise we could extend the maximal set `J`
    have hcard : (insert e I₀).ncard ≤ J.ncard := by
      refine not_lt.1 fun hlt ↦ ?_
      obtain ⟨f, hfI, hfJ, hfi⟩ := indep_aug hJ hJfin heI₀i (hI₀fin.insert e) hlt
      have hfE₀ : f ∈ E₀ := mem_of_mem_of_subset hfI (insert_subset_insert subset_union_left)
      refine hfJ (insert_eq_self.1 <| Eq.symm (hJmax _
        ⟨hB₀J.trans <| subset_insert _ _,hfi,insert_subset hfE₀ hJss⟩ (subset_insert _ _)))

    -- But this means `|I₀| < |J|`, and extending `I₀` into `J` gives a contradiction
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      heI₀ : Not (Membership.mem I₀ e)
      heI₀i : Indep (Insert.insert e I₀)
      heJ : Not (Membership.mem J e)
      hJfin : J.Finite
      hcard : LE.le (Insert.insert e I₀).ncard J.ncard
      ⊢ False
    -/
    rw [ncard_insert_of_not_mem heI₀ hI₀fin, ← Nat.lt_iff_add_one_le] at hcard

    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      heI₀ : Not (Membership.mem I₀ e)
      heI₀i : Indep (Insert.insert e I₀)
      heJ : Not (Membership.mem J e)
      hJfin : J.Finite
      hcard : LT.lt I₀.ncard J.ncard
      ⊢ False
    -/
    obtain ⟨f, hfJ, hfI₀, hfi⟩ := indep_aug (indep_subset hI hI₀I) hI₀fin hJ hJfin hcard
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.int …
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      I B : Set α
      hI : Indep I
      hImax : Not (Maximal Indep I)
      hBmax : Maximal Indep B
      e : α
      heI : Not (Membership.mem I e)
      hins : Indep (Insert.insert e I)
      heB : Not (Membership.mem B e)
      hcon : ∀ (x : α), Membership.mem (SDiff.sdiff B I) x → Not (Indep (Insert.inse …
      heBdep : Not (Indep (Insert.insert e B))
      B₀ : Set α
      hB₀B : HasSubset.Subset B₀ B
      hB₀fin : B₀.Finite
      hB₀e : Not (Indep (Insert.insert e B₀))
      hB₀ : Indep B₀
      I₀ : Set α
      hI₀I : HasSubset.Subset I₀ I
      hI₀fin : I₀.Finite
      hI₀ : ∀ (x : α), Membership.mem (SDiff.sdiff B₀ I₀) x → Not (Indep (Insert.ins …
      E₀ : Set α := Insert.insert e (Union.union I₀ B₀)
      hE₀fin : E₀.Finite
      J : Set α
      hJmax : ∀ (a' : Set α), Membership.mem (setOf fun J => And (HasSubset.Subset B …
      hB₀J : HasSubset.Subset B₀ J
      hJ : Indep J
      hJss : HasSubset.Subset J E₀
      heI₀ : Not (Membership.mem I₀ e)
      heI₀i : Indep (Insert.insert e I₀)
      heJ : Not (Membership.mem J e)
      hJfin : J.Finite
      hcard : LT.lt I₀.ncard J.ncard
      f : α
      hfJ : Membership.mem J f
      hfI₀ : Not (Membership.mem I₀ f)
      hfi : Indep (Insert.insert f I₀)
      ⊢ False
    -/
    exact hI₀ f ⟨Or.elim (hJss hfJ) (fun hfe ↦ (heJ <| hfe ▸ hfJ).elim) (by aesop), hfI₀⟩ hfi)
    /-
      🎉 no goals
    -/
  (indep_maximal := by
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      ⊢ ∀ (X : Set α), HasSubset.Subset X E → Matroid.ExistsMaximalSubsetProperty In …
    -/
    refine fun X _ I hI hIX ↦ zorn_subset_nonempty {Y | Indep Y ∧ Y ⊆ X} ?_ I ⟨hI, hIX⟩
    refine fun Is hIs hchain _ ↦
      ⟨⋃₀ Is, ⟨?_, sUnion_subset fun Y hY ↦ (hIs hY).2⟩, fun _ ↦ subset_sUnion_of_mem⟩
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      ⊢ Indep Is.sUnion
    -/
    refine indep_compact _ fun J hJ hJfin ↦ ?_
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      ⊢ Indep J
    -/
    have hchoose : ∀ e, e ∈ J → ∃ I, I ∈ Is ∧ (e : α) ∈ I := fun _ he ↦ mem_sUnion.1 <| hJ he
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      hchoose : ∀ (e : α), Membership.mem J e → Exists fun I => And (Membership.mem  …
      ⊢ Indep J
    -/
    choose! f hf using hchoose
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      f : α → Set α
      hf : ∀ (e : α), Membership.mem J e → And (Membership.mem Is (f e)) (Membership …
      ⊢ Indep J
    -/
    refine J.eq_empty_or_nonempty.elim (fun hJ ↦ hJ ▸ indep_empty) (fun hne ↦ ?_)
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      f : α → Set α
      hf : ∀ (e : α), Membership.mem J e → And (Membership.mem Is (f e)) (Membership …
      hne : J.Nonempty
      ⊢ Indep J
    -/
    obtain ⟨x, hxJ, hxmax⟩ := Finite.exists_maximal_wrt f _ hJfin hne
    /-
      case intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      f : α → Set α
      hf : ∀ (e : α), Membership.mem J e → And (Membership.mem Is (f e)) (Membership …
      hne : J.Nonempty
      x : α
      hxJ : Membership.mem J x
      hxmax : ∀ (a' : α), Membership.mem J a' → LE.le (f x) (f a') → Eq (f x) (f a')
      ⊢ Indep J
    -/
    refine indep_subset (hIs (hf x hxJ).1).1 fun y hyJ ↦ ?_
    /-
      case intro.intro
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      f : α → Set α
      hf : ∀ (e : α), Membership.mem J e → And (Membership.mem Is (f e)) (Membership …
      hne : J.Nonempty
      x : α
      hxJ : Membership.mem J x
      hxmax : ∀ (a' : α), Membership.mem J a' → LE.le (f x) (f a') → Eq (f x) (f a')
      y : α
      hyJ : Membership.mem J y
      ⊢ Membership.mem (f x) y
    -/
    obtain (hle | hle) := hchain.total (hf _ hxJ).1 (hf _ hyJ).1
      /-
        case intro.intro.inl
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
        X : Set α
        x✝¹ : HasSubset.Subset X E
        I : Set α
        hI : Indep I
        hIX : HasSubset.Subset I X
        Is : Set (Set α)
        hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
        hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
        x✝ : Is.Nonempty
        J : Set α
        hJ : HasSubset.Subset J Is.sUnion
        hJfin : J.Finite
        f : α → Set α
        hf : ∀ (e : α), Membership.mem J e → And (Membership.mem Is (f e)) (Membership …
        hne : J.Nonempty
        x : α
        hxJ : Membership.mem J x
        hxmax : ∀ (a' : α), Membership.mem J a' → LE.le (f x) (f a') → Eq (f x) (f a')
        y : α
        hyJ : Membership.mem J y
        hle : HasSubset.Subset (f x) (f y)
        ⊢ Membership.mem (f x) y
      -/
    · rw [hxmax _ hyJ hle]; exact (hf _ hyJ).2
                            /-
                              🎉 no goals
                            -/
    /-
      case intro.intro.inr
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      indep_empty : Indep EmptyCollection.emptyCollection
      indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
      indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
      indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
      subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
      htofin : ∀ (I : Set α) (e : α), Indep I → Not (Indep (Insert.insert e I)) → Ex …
      X : Set α
      x✝¹ : HasSubset.Subset X E
      I : Set α
      hI : Indep I
      hIX : HasSubset.Subset I X
      Is : Set (Set α)
      hIs : HasSubset.Subset Is (setOf fun Y => And (Indep Y) (HasSubset.Subset Y X))
      hchain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) Is
      x✝ : Is.Nonempty
      J : Set α
      hJ : HasSubset.Subset J Is.sUnion
      hJfin : J.Finite
      f : α → Set α
      hf : ∀ (e : α), Membership.mem J e → And (Membership.mem Is (f e)) (Membership …
      hne : J.Nonempty
      x : α
      hxJ : Membership.mem J x
      hxmax : ∀ (a' : α), Membership.mem J a' → LE.le (f x) (f a') → Eq (f x) (f a')
      y : α
      hyJ : Membership.mem J y
      hle : HasSubset.Subset (f y) (f x)
      ⊢ Membership.mem (f x) y
    -/
    exact hle (hf _ hyJ).2)
    /-
      🎉 no goals
    -/

  (subset_ground := subset_ground)


@[simp] theorem ofFinitary_indep (E : Set α) (Indep : Set α → Prop)
    indep_empty indep_subset indep_aug indep_compact subset_ground : (IndepMatroid.ofFinitary
      E Indep indep_empty indep_subset indep_aug indep_compact subset_ground).Indep = Indep := rfl


instance ofFinitary_finitary (E : Set α) (Indep : Set α → Prop)
    indep_empty indep_subset indep_aug indep_compact subset_ground : Finitary
    (IndepMatroid.ofFinitary
      E Indep indep_empty indep_subset indep_aug indep_compact subset_ground).matroid :=
      /-
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → I.Finite → Indep J → J.Finite → LT.lt I …
        indep_compact : ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite …
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        ⊢ ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite → (IndepMatro …
      -/
  ⟨by simpa⟩
      /-
        🎉 no goals
      -/


/-- If there is an absolute upper bound on the size of a set satisfying `P`, then the
  maximal subset property always holds. -/
theorem _root_.Matroid.existsMaximalSubsetProperty_of_bdd {P : Set α → Prop}
    (hP : ∃ (n : ℕ), ∀ Y, P Y → Y.encard ≤ n) (X : Set α) : ExistsMaximalSubsetProperty P X := by
  /-
    α : Type u_1
    P : Set α → Prop
    hP : Exists fun n => ∀ (Y : Set α), P Y → LE.le Y.encard ↑n
    X : Set α
    ⊢ Matroid.ExistsMaximalSubsetProperty P X
  -/
  obtain ⟨n, hP⟩ := hP
  /-
    case intro
    α : Type u_1
    P : Set α → Prop
    X : Set α
    n : Nat
    hP : ∀ (Y : Set α), P Y → LE.le Y.encard ↑n
    ⊢ Matroid.ExistsMaximalSubsetProperty P X
  -/
  rintro I hI hIX
  have hfin : Set.Finite (ncard '' {Y | P Y ∧ I ⊆ Y ∧ Y ⊆ X}) := by
    rw [finite_iff_bddAbove, bddAbove_def]
    simp_rw [ENat.le_coe_iff] at hP
    use n
    rintro x ⟨Y, ⟨hY,-,-⟩, rfl⟩
    obtain ⟨n₀, heq, hle⟩ := hP Y hY
    rwa [ncard_def, heq, ENat.toNat_coe]
  obtain ⟨Y, ⟨hY, hIY, hYX⟩, hY'⟩ :=
    Finite.exists_maximal_wrt' ncard _ hfin ⟨I, hI, rfl.subset, hIX⟩

  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    P : Set α → Prop
    X : Set α
    n : Nat
    hP : ∀ (Y : Set α), P Y → LE.le Y.encard ↑n
    I : Set α
    hI : P I
    hIX : HasSubset.Subset I X
    hfin : (Set.image Set.ncard (setOf fun Y => And (P Y) (And (HasSubset.Subset I …
    Y : Set α
    hY' : ∀ (a' : Set α), Membership.mem (setOf fun Y => And (P Y) (And (HasSubset …
    hY : P Y
    hIY : HasSubset.Subset I Y
    hYX : HasSubset.Subset Y X
    ⊢ Exists fun J => And (HasSubset.Subset I J) (Maximal (fun K => And (P K) (Has …
  -/
  refine ⟨Y, hIY, ⟨hY, hYX⟩, fun K ⟨hPK, hKX⟩ hYK ↦ ?_⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    P : Set α → Prop
    X : Set α
    n : Nat
    hP : ∀ (Y : Set α), P Y → LE.le Y.encard ↑n
    I : Set α
    hI : P I
    hIX : HasSubset.Subset I X
    hfin : (Set.image Set.ncard (setOf fun Y => And (P Y) (And (HasSubset.Subset I …
    Y : Set α
    hY' : ∀ (a' : Set α), Membership.mem (setOf fun Y => And (P Y) (And (HasSubset …
    hY : P Y
    hIY : HasSubset.Subset I Y
    hYX : HasSubset.Subset Y X
    K : Set α
    x✝ : (fun K => And (P K) (HasSubset.Subset K X)) K
    hYK : LE.le Y K
    hPK : P K
    hKX : HasSubset.Subset K X
    ⊢ LE.le K Y
  -/
  have hKfin : K.Finite := finite_of_encard_le_coe (hP K hPK)

  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    P : Set α → Prop
    X : Set α
    n : Nat
    hP : ∀ (Y : Set α), P Y → LE.le Y.encard ↑n
    I : Set α
    hI : P I
    hIX : HasSubset.Subset I X
    hfin : (Set.image Set.ncard (setOf fun Y => And (P Y) (And (HasSubset.Subset I …
    Y : Set α
    hY' : ∀ (a' : Set α), Membership.mem (setOf fun Y => And (P Y) (And (HasSubset …
    hY : P Y
    hIY : HasSubset.Subset I Y
    hYX : HasSubset.Subset Y X
    K : Set α
    x✝ : (fun K => And (P K) (HasSubset.Subset K X)) K
    hYK : LE.le Y K
    hPK : P K
    hKX : HasSubset.Subset K X
    hKfin : K.Finite
    ⊢ LE.le K Y
  -/
  refine (eq_of_subset_of_ncard_le hYK ?_ hKfin).symm.subset
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    P : Set α → Prop
    X : Set α
    n : Nat
    hP : ∀ (Y : Set α), P Y → LE.le Y.encard ↑n
    I : Set α
    hI : P I
    hIX : HasSubset.Subset I X
    hfin : (Set.image Set.ncard (setOf fun Y => And (P Y) (And (HasSubset.Subset I …
    Y : Set α
    hY' : ∀ (a' : Set α), Membership.mem (setOf fun Y => And (P Y) (And (HasSubset …
    hY : P Y
    hIY : HasSubset.Subset I Y
    hYX : HasSubset.Subset Y X
    K : Set α
    x✝ : (fun K => And (P K) (HasSubset.Subset K X)) K
    hYK : LE.le Y K
    hPK : P K
    hKX : HasSubset.Subset K X
    hKfin : K.Finite
    ⊢ LE.le K.ncard Y.ncard
  -/
  rw [hY' K ⟨hPK, hIY.trans hYK, hKX⟩ (ncard_le_ncard hYK hKfin)]
  /-
    🎉 no goals
  -/


/-- If there is an absolute upper bound on the size of an independent set, then the maximality axiom
  isn't needed to define a matroid by independent sets. -/
@[simps E] protected def ofBdd (E : Set α) (Indep : Set α → Prop)
    (indep_empty : Indep ∅)
    (indep_subset : ∀ ⦃I J⦄, Indep J → I ⊆ J → Indep I)
    (indep_aug : ∀⦃I B⦄, Indep I → ¬ Maximal Indep I → Maximal Indep B →
      ∃ x ∈ B \ I, Indep (insert x I))
    (subset_ground : ∀ I, Indep I → I ⊆ E)
    (indep_bdd : ∃ (n : ℕ), ∀ I, Indep I → I.encard ≤ n ) : IndepMatroid α where
  E := E
  Indep := Indep
  indep_empty := indep_empty
  indep_subset := indep_subset
  indep_aug := indep_aug
  indep_maximal X _ := Matroid.existsMaximalSubsetProperty_of_bdd indep_bdd X
  subset_ground := subset_ground


@[simp] theorem ofBdd_indep (E : Set α) Indep indep_empty indep_subset indep_aug
    subset_ground h_bdd : (IndepMatroid.ofBdd
      E Indep indep_empty indep_subset indep_aug subset_ground h_bdd).Indep = Indep := rfl


/-- `IndepMatroid.ofBdd` constructs a `FiniteRk` matroid. -/
instance (E : Set α) (Indep : Set α → Prop) indep_empty indep_subset indep_aug subset_ground h_bdd :
    FiniteRk (IndepMatroid.ofBdd
      E Indep indep_empty indep_subset indep_aug subset_ground h_bdd).matroid := by
  /-
    α : Type u_1
    E : Set α
    Indep : Set α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B …
    subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
    h_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
    ⊢ (IndepMatroid.ofBdd E Indep indep_empty indep_subset indep_aug subset_ground …
  -/
  obtain ⟨B, hB⟩ := (IndepMatroid.ofBdd E Indep _ _ _ _ _).matroid.exists_base
  /-
    case intro
    α : Type u_1
    E : Set α
    Indep : Set α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B …
    subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
    h_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
    B : Set α
    hB : (IndepMatroid.ofBdd E Indep ?m.262105 ?m.262106 ?m.262107 ?m.262108 ?m.26 …
    ⊢ (IndepMatroid.ofBdd E Indep indep_empty indep_subset indep_aug subset_ground …
  -/
  refine hB.finiteRk_of_finite ?_
  /-
    case intro
    α : Type u_1
    E : Set α
    Indep : Set α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B …
    subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
    h_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
    B : Set α
    hB : (IndepMatroid.ofBdd E Indep indep_empty indep_subset indep_aug subset_gro …
    ⊢ B.Finite
  -/
  obtain ⟨n, hn⟩ := h_bdd
  /-
    case intro.intro
    α : Type u_1
    E : Set α
    Indep : Set α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B …
    subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
    B : Set α
    n : Nat
    hn : ∀ (I : Set α), Indep I → LE.le I.encard ↑n
    hB : (IndepMatroid.ofBdd E Indep indep_empty indep_subset indep_aug subset_gro …
    ⊢ B.Finite
  -/
  exact finite_of_encard_le_coe <| hn B (by simpa using hB.indep)
  /-
    🎉 no goals
  -/


/-- If there is an absolute upper bound on the size of an independent set, then matroids
  can be defined using an 'augmentation' axiom similar to the standard definition of
  finite matroids for independent sets. -/
protected def ofBddAugment (E : Set α) (Indep : Set α → Prop)
    (indep_empty : Indep ∅)
    (indep_subset : ∀ ⦃I J⦄, Indep J → I ⊆ J → Indep I)
    (indep_aug : ∀ ⦃I J⦄, Indep I → Indep J → I.encard < J.encard →
      ∃ e ∈ J, e ∉ I ∧ Indep (insert e I))
    (indep_bdd : ∃ (n : ℕ), ∀ I, Indep I → I.encard ≤ n )
    (subset_ground : ∀ I, Indep I → I ⊆ E) : IndepMatroid α :=
  IndepMatroid.ofBdd (E := E) (Indep := Indep)
    (indep_empty := indep_empty)
    (indep_subset := indep_subset)
    (indep_aug := by
      /-
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
        indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        ⊢ ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B → Exists  …
      -/
      rintro I B hI hImax hBmax
      suffices hcard : I.encard < B.encard by
        obtain ⟨e, heB, heI, hi⟩ := indep_aug hI hBmax.prop hcard
        exact ⟨e, ⟨heB, heI⟩, hi⟩
      /-
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
        indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        I B : Set α
        hI : Indep I
        hImax : Not (Maximal Indep I)
        hBmax : Maximal Indep B
        ⊢ LT.lt I.encard B.encard
      -/
      refine lt_of_not_le fun hle ↦ ?_
      /-
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
        indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        I B : Set α
        hI : Indep I
        hImax : Not (Maximal Indep I)
        hBmax : Maximal Indep B
        hle : LE.le B.encard I.encard
        ⊢ False
      -/
      obtain ⟨x, hxnot, hxI⟩ := exists_insert_of_not_maximal indep_subset hI hImax
      have hlt : B.encard < (insert x I).encard := by
        rwa [encard_insert_of_not_mem hxnot, ← not_le, ENat.add_one_le_iff, not_lt]
        rw [encard_ne_top_iff]
        obtain ⟨n, hn⟩ := indep_bdd
        exact finite_of_encard_le_coe (hn _ hI)
      /-
        case intro.intro
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
        indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        I B : Set α
        hI : Indep I
        hImax : Not (Maximal Indep I)
        hBmax : Maximal Indep B
        hle : LE.le B.encard I.encard
        x : α
        hxnot : Not (Membership.mem I x)
        hxI : Indep (Insert.insert x I)
        hlt : LT.lt B.encard (Insert.insert x I).encard
        ⊢ False
      -/
      obtain ⟨y, -, hyB, hi⟩ := indep_aug hBmax.prop hxI hlt
      /-
        case intro.intro.intro.intro.intro
        α : Type u_1
        E : Set α
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
        indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
        subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
        I B : Set α
        hI : Indep I
        hImax : Not (Maximal Indep I)
        hBmax : Maximal Indep B
        hle : LE.le B.encard I.encard
        x : α
        hxnot : Not (Membership.mem I x)
        hxI : Indep (Insert.insert x I)
        hlt : LT.lt B.encard (Insert.insert x I).encard
        y : α
        hyB : Not (Membership.mem B y)
        hi : Indep (Insert.insert y B)
        ⊢ False
      -/
      exact hBmax.not_prop_of_ssuperset (ssubset_insert hyB) hi)
      /-
        🎉 no goals
      -/
    (indep_bdd := indep_bdd) (subset_ground := subset_ground)


@[simp] theorem ofBddAugment_E (E : Set α) Indep indep_empty indep_subset indep_aug
    indep_bdd subset_ground : (IndepMatroid.ofBddAugment
      E Indep indep_empty indep_subset indep_aug indep_bdd subset_ground).E = E := rfl


@[simp] theorem ofBddAugment_indep (E : Set α) Indep indep_empty indep_subset indep_aug
    indep_bdd subset_ground : (IndepMatroid.ofBddAugment
      E Indep indep_empty indep_subset indep_aug indep_bdd subset_ground).Indep = Indep := rfl


instance ofBddAugment_finiteRk (E : Set α) Indep indep_empty indep_subset indep_aug
    indep_bdd subset_ground : FiniteRk (IndepMatroid.ofBddAugment
      E Indep indep_empty indep_subset indep_aug indep_bdd subset_ground).matroid := by
  /-
    α : Type u_1
    E : Set α
    Indep : Set α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
    indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
    subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
    ⊢ (IndepMatroid.ofBddAugment E Indep indep_empty indep_subset indep_aug indep_ …
  -/
  rw [IndepMatroid.ofBddAugment]
  /-
    α : Type u_1
    E : Set α
    Indep : Set α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exi …
    indep_bdd : Exists fun n => ∀ (I : Set α), Indep I → LE.le I.encard ↑n
    subset_ground : ∀ (I : Set α), Indep I → HasSubset.Subset I E
    ⊢ (IndepMatroid.ofBdd E Indep indep_empty indep_subset ⋯ subset_ground indep_b …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `E` is finite, then any collection of subsets of `E` satisfying
  the usual independence axioms determines a matroid -/
protected def ofFinite {E : Set α} (hE : E.Finite) (Indep : Set α → Prop)
    (indep_empty : Indep ∅)
    (indep_subset : ∀ ⦃I J⦄, Indep J → I ⊆ J → Indep I)
    (indep_aug :
      ∀ ⦃I J⦄, Indep I → Indep J → I.ncard < J.ncard → ∃ e ∈ J, e ∉ I ∧ Indep (insert e I))
    (subset_ground : ∀ ⦃I⦄, Indep I → I ⊆ E) : IndepMatroid α :=
  IndepMatroid.ofBddAugment (E := E) (Indep := Indep) (indep_empty := indep_empty)
    (indep_subset := indep_subset)
    (indep_aug := by
      /-
        α : Type u_1
        E : Set α
        hE : E.Finite
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.ncard J.ncard → Exist …
        subset_ground : ∀ ⦃I : Set α⦄, Indep I → HasSubset.Subset I E
        ⊢ ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.encard J.encard → Exists fun e  …
      -/
      refine fun {I J} hI hJ hIJ ↦ indep_aug hI hJ ?_
      rwa [← Nat.cast_lt (α := ℕ∞), (hE.subset (subset_ground hJ)).cast_ncard_eq,
        (hE.subset (subset_ground hI)).cast_ncard_eq] )
    (indep_bdd := ⟨E.ncard, fun I hI ↦ by
      /-
        α : Type u_1
        E : Set α
        hE : E.Finite
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.ncard J.ncard → Exist …
        subset_ground : ∀ ⦃I : Set α⦄, Indep I → HasSubset.Subset I E
        I : Set α
        hI : Indep I
        ⊢ LE.le I.encard ↑E.ncard
      -/
      rw [hE.cast_ncard_eq]
      /-
        α : Type u_1
        E : Set α
        hE : E.Finite
        Indep : Set α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Set α⦄, Indep I → Indep J → LT.lt I.ncard J.ncard → Exist …
        subset_ground : ∀ ⦃I : Set α⦄, Indep I → HasSubset.Subset I E
        I : Set α
        hI : Indep I
        ⊢ LE.le I.encard E.encard
      -/
      exact encard_le_card <| subset_ground hI ⟩)
      /-
        🎉 no goals
      -/
    (subset_ground := subset_ground)


@[simp] theorem ofFinite_E {E : Set α} hE Indep indep_empty indep_subset indep_aug subset_ground :
    (IndepMatroid.ofFinite
      (hE : E.Finite) Indep indep_empty indep_subset indep_aug subset_ground).E = E := rfl


@[simp] theorem ofFinite_indep {E : Set α} hE Indep indep_empty indep_subset indep_aug
    subset_ground : (IndepMatroid.ofFinite
      (hE : E.Finite) Indep indep_empty indep_subset indep_aug subset_ground).Indep = Indep := rfl


instance ofFinite_finite {E : Set α} hE Indep indep_empty indep_subset indep_aug subset_ground :
    (IndepMatroid.ofFinite
      (hE : E.Finite) Indep indep_empty indep_subset indep_aug subset_ground).matroid.Finite :=
  ⟨hE⟩


/-- An independence predicate on `Finset α` that obeys the finite matroid axioms determines a
  finitary matroid on `α`. -/
protected def ofFinset [DecidableEq α] (E : Set α) (Indep : Finset α → Prop)
    (indep_empty : Indep ∅)
    (indep_subset : ∀ ⦃I J⦄, Indep J → I ⊆ J → Indep I)
    (indep_aug : ∀ ⦃I J⦄, Indep I → Indep J → I.card < J.card → ∃ e ∈ J, e ∉ I ∧ Indep (insert e I))
    (subset_ground : ∀ ⦃I⦄, Indep I → (I : Set α) ⊆ E) : IndepMatroid α :=
  IndepMatroid.ofFinitary
    (E := E)
    (Indep := (fun I ↦ (∀ (J : Finset α), (J : Set α) ⊆ I → Indep J)))
                       /-
                         α : Type u_1
                         inst✝ : DecidableEq α
                         E : Set α
                         Indep : Finset α → Prop
                         indep_empty : Indep EmptyCollection.emptyCollection
                         indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
                         indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
                         subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
                         ⊢ (fun I => ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J) EmptyCollecti …
                       -/
    (indep_empty := by simpa [subset_empty_iff])
                       /-
                         🎉 no goals
                       -/
    (indep_subset := ( fun _ _ hJ hIJ _ hKI ↦ hJ _ (hKI.trans hIJ) ))
    (indep_aug := by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        E : Set α
        Indep : Finset α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
        subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
        ⊢ ∀ ⦃I J : Set α⦄, (fun I => ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep …
      -/
      intro I J hI hIfin hJ hJfin hIJ
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        E : Set α
        Indep : Finset α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
        subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
        I J : Set α
        hI : ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J
        hIfin : I.Finite
        hJ : ∀ (J_1 : Finset α), HasSubset.Subset (↑J_1) J → Indep J_1
        hJfin : J.Finite
        hIJ : LT.lt I.ncard J.ncard
        ⊢ Exists fun e => And (Membership.mem J e) (And (Not (Membership.mem I e)) ((f …
      -/
      rw [ncard_eq_toFinset_card _ hIfin, ncard_eq_toFinset_card _ hJfin] at hIJ
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        E : Set α
        Indep : Finset α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
        subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
        I J : Set α
        hI : ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J
        hIfin : I.Finite
        hJ : ∀ (J_1 : Finset α), HasSubset.Subset (↑J_1) J → Indep J_1
        hJfin : J.Finite
        hIJ : LT.lt hIfin.toFinset.card hJfin.toFinset.card
        ⊢ Exists fun e => And (Membership.mem J e) (And (Not (Membership.mem I e)) ((f …
      -/
      have aug := indep_aug (hI _ (by simp [Subset.rfl])) (hJ _ (by simp [Subset.rfl])) hIJ
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        E : Set α
        Indep : Finset α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
        subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
        I J : Set α
        hI : ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J
        hIfin : I.Finite
        hJ : ∀ (J_1 : Finset α), HasSubset.Subset (↑J_1) J → Indep J_1
        hJfin : J.Finite
        hIJ : LT.lt hIfin.toFinset.card hJfin.toFinset.card
        aug : Exists fun e => And (Membership.mem hJfin.toFinset e) (And (Not (Members …
        ⊢ Exists fun e => And (Membership.mem J e) (And (Not (Membership.mem I e)) ((f …
      -/
      simp only [Finite.mem_toFinset] at aug
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        E : Set α
        Indep : Finset α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
        subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
        I J : Set α
        hI : ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J
        hIfin : I.Finite
        hJ : ∀ (J_1 : Finset α), HasSubset.Subset (↑J_1) J → Indep J_1
        hJfin : J.Finite
        hIJ : LT.lt hIfin.toFinset.card hJfin.toFinset.card
        aug : Exists fun e => And (Membership.mem J e) (And (Not (Membership.mem I e)) …
        ⊢ Exists fun e => And (Membership.mem J e) (And (Not (Membership.mem I e)) ((f …
      -/
      obtain ⟨e, heJ, heI, hi⟩ := aug
      /-
        case intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        E : Set α
        Indep : Finset α → Prop
        indep_empty : Indep EmptyCollection.emptyCollection
        indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
        indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
        subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
        I J : Set α
        hI : ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J
        hIfin : I.Finite
        hJ : ∀ (J_1 : Finset α), HasSubset.Subset (↑J_1) J → Indep J_1
        hJfin : J.Finite
        hIJ : LT.lt hIfin.toFinset.card hJfin.toFinset.card
        e : α
        heJ : Membership.mem J e
        heI : Not (Membership.mem I e)
        hi : Indep (Insert.insert e hIfin.toFinset)
        ⊢ Exists fun e => And (Membership.mem J e) (And (Not (Membership.mem I e)) ((f …
      -/
      exact ⟨e, heJ, heI, fun K hK ↦ indep_subset hi <| Finset.coe_subset.1 (by simpa)⟩ )
      /-
        🎉 no goals
      -/
    (indep_compact := fun _ h J hJ ↦ h _ hJ J.finite_toSet _ Subset.rfl )
                                          /-
                                            α : Type u_1
                                            inst✝ : DecidableEq α
                                            E : Set α
                                            Indep : Finset α → Prop
                                            indep_empty : Indep EmptyCollection.emptyCollection
                                            indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
                                            indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
                                            subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
                                            I : Set α
                                            hI : (fun I => ∀ (J : Finset α), HasSubset.Subset (↑J) I → Indep J) I
                                            x : α
                                            hxI : Membership.mem I x
                                            ⊢ Membership.mem E x
                                          -/
    (subset_ground := fun I hI x hxI ↦ by simpa using subset_ground <| hI {x} (by simpa) )
                                          /-
                                            🎉 no goals
                                          -/


@[simp] theorem ofFinset_E [DecidableEq α] (E : Set α) Indep indep_empty indep_subset indep_aug
    subset_ground : (IndepMatroid.ofFinset
      E Indep indep_empty indep_subset indep_aug subset_ground).E = E := rfl


@[simp] theorem ofFinset_indep [DecidableEq α] (E : Set α) Indep indep_empty indep_subset indep_aug
    subset_ground {I : Finset α} : (IndepMatroid.ofFinset
      E Indep indep_empty indep_subset indep_aug subset_ground).Indep I ↔ Indep I := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    E : Set α
    Indep : Finset α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
    subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
    I : Finset α
    ⊢ Iff ((IndepMatroid.ofFinset E Indep indep_empty indep_subset indep_aug subse …
  -/
  simp only [IndepMatroid.ofFinset, ofFinitary_indep, Finset.coe_subset]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    E : Set α
    Indep : Finset α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
    subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
    I : Finset α
    ⊢ Iff (∀ (J : Finset α), HasSubset.Subset J I → Indep J) (Indep I)
  -/
  exact ⟨fun h ↦ h _ Subset.rfl, fun h J hJI ↦ indep_subset h hJI⟩
  /-
    🎉 no goals
  -/


/-- This can't be `@[simp]`, because it would cause the more useful
  `Matroid.ofIndepFinset_apply` not to be in simp normal form. -/
theorem ofFinset_indep' [DecidableEq α] (E : Set α) Indep indep_empty indep_subset indep_aug
    subset_ground {I : Set α} : (IndepMatroid.ofFinset
      E Indep indep_empty indep_subset indep_aug subset_ground).Indep I ↔
        ∀ (J : Finset α), (J : Set α) ⊆ I → Indep J := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    E : Set α
    Indep : Finset α → Prop
    indep_empty : Indep EmptyCollection.emptyCollection
    indep_subset : ∀ ⦃I J : Finset α⦄, Indep J → HasSubset.Subset I J → Indep I
    indep_aug : ∀ ⦃I J : Finset α⦄, Indep I → Indep J → LT.lt I.card J.card → Exis …
    subset_ground : ∀ ⦃I : Finset α⦄, Indep I → HasSubset.Subset (↑I) E
    I : Set α
    ⊢ Iff ((IndepMatroid.ofFinset E Indep indep_empty indep_subset indep_aug subse …
  -/
  simp only [IndepMatroid.ofFinset, ofFinitary_indep]
  /-
    🎉 no goals
  -/


/-- Construct an `Matroid` from an independence predicate that agrees with that of some matroid `M`.
  This is computable even if `M` is only known existentially, or when `M` exists for different
  reasons in different cases. This can also be used to change the independence predicate to a
  more useful definitional form. -/
@[simps! E] protected def ofExistsMatroid (E : Set α) (Indep : Set α → Prop)
    (hM : ∃ (M : Matroid α), E = M.E ∧ ∀ I, M.Indep I ↔ Indep I) : Matroid α :=
  IndepMatroid.matroid <|
  have hex : ∃ (M : Matroid α), E = M.E ∧ M.Indep = Indep := by
    /-
      α : Type u_1
      E : Set α
      Indep : Set α → Prop
      hM : Exists fun M => And (Eq E M.E) (∀ (I : Set α), Iff (M.Indep I) (Indep I))
      ⊢ Exists fun M => And (Eq E M.E) (Eq M.Indep Indep)
    -/
    obtain ⟨M, rfl, h⟩ := hM; refine ⟨_, rfl, funext (by simp [h])⟩
                              /-
                                🎉 no goals
                              -/
  IndepMatroid.mk (E := E) (Indep := Indep)
                     /-
                       α : Type u_1
                       E : Set α
                       Indep : Set α → Prop
                       hM : Exists fun M => And (Eq E M.E) (∀ (I : Set α), Iff (M.Indep I) (Indep I))
                       hex : Exists fun M => And (Eq E M.E) (Eq M.Indep Indep)
                       ⊢ Indep EmptyCollection.emptyCollection
                     -/
  (indep_empty := by obtain ⟨M, -, rfl⟩ := hex; exact M.empty_indep)
                                                /-
                                                  🎉 no goals
                                                -/
                      /-
                        α : Type u_1
                        E : Set α
                        Indep : Set α → Prop
                        hM : Exists fun M => And (Eq E M.E) (∀ (I : Set α), Iff (M.Indep I) (Indep I))
                        hex : Exists fun M => And (Eq E M.E) (Eq M.Indep Indep)
                        ⊢ ∀ ⦃I J : Set α⦄, Indep J → HasSubset.Subset I J → Indep I
                      -/
  (indep_subset := by obtain ⟨M, -, rfl⟩ := hex; exact fun I J hJ hIJ ↦ hJ.subset hIJ)
                                                 /-
                                                   🎉 no goals
                                                 -/
                   /-
                     α : Type u_1
                     E : Set α
                     Indep : Set α → Prop
                     hM : Exists fun M => And (Eq E M.E) (∀ (I : Set α), Iff (M.Indep I) (Indep I))
                     hex : Exists fun M => And (Eq E M.E) (Eq M.Indep Indep)
                     ⊢ ∀ ⦃I B : Set α⦄, Indep I → Not (Maximal Indep I) → Maximal Indep B → Exists  …
                   -/
  (indep_aug := by obtain ⟨M, -, rfl⟩ := hex; exact Indep.exists_insert_of_not_maximal M)
                                              /-
                                                🎉 no goals
                                              -/
                       /-
                         α : Type u_1
                         E : Set α
                         Indep : Set α → Prop
                         hM : Exists fun M => And (Eq E M.E) (∀ (I : Set α), Iff (M.Indep I) (Indep I))
                         hex : Exists fun M => And (Eq E M.E) (Eq M.Indep Indep)
                         ⊢ ∀ (X : Set α), HasSubset.Subset X E → Matroid.ExistsMaximalSubsetProperty In …
                       -/
  (indep_maximal := by obtain ⟨M, rfl, rfl⟩ := hex; exact M.existsMaximalSubsetProperty_indep)
                                                    /-
                                                      🎉 no goals
                                                    -/
                       /-
                         α : Type u_1
                         E : Set α
                         Indep : Set α → Prop
                         hM : Exists fun M => And (Eq E M.E) (∀ (I : Set α), Iff (M.Indep I) (Indep I))
                         hex : Exists fun M => And (Eq E M.E) (Eq M.Indep Indep)
                         ⊢ ∀ (I : Set α), Indep I → HasSubset.Subset I E
                       -/
  (subset_ground := by obtain ⟨M, rfl, rfl⟩ := hex; exact fun I ↦ Indep.subset_ground)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- A matroid defined purely in terms of its bases. -/
@[simps E] protected def ofBase (E : Set α) (Base : Set α → Prop) (exists_base : ∃ B, Base B)
    (base_exchange : ExchangeProperty Base)
    (maximality : ∀ X, X ⊆ E → Matroid.ExistsMaximalSubsetProperty (∃ B, Base B ∧ · ⊆ B) X)
    (subset_ground : ∀ B, Base B → B ⊆ E) : Matroid α where
  E := E
  Base := Base
  Indep I := (∃ B, Base B ∧ I ⊆ B)
  indep_iff' _ := Iff.rfl
  exists_base := exists_base
  base_exchange := base_exchange
  maximality := maximality
  subset_ground := subset_ground


/-- A collection of bases with the exchange property and at least one finite member is a matroid -/
@[simps! E] protected def ofExistsFiniteBase (E : Set α) (Base : Set α → Prop)
    (exists_finite_base : ∃ B, Base B ∧ B.Finite) (base_exchange : ExchangeProperty Base)
    (subset_ground : ∀ B, Base B → B ⊆ E) : Matroid α := Matroid.ofBase
  (E := E)
  (Base := Base)
                     /-
                       α : Type u_1
                       E : Set α
                       Base : Set α → Prop
                       exists_finite_base : Exists fun B => And (Base B) B.Finite
                       base_exchange : Matroid.ExchangeProperty Base
                       subset_ground : ∀ (B : Set α), Base B → HasSubset.Subset B E
                       ⊢ Exists fun B => Base B
                     -/
  (exists_base := by obtain ⟨B,h⟩ := exists_finite_base; exact ⟨B, h.1⟩)
                                                         /-
                                                           🎉 no goals
                                                         -/
  (base_exchange := base_exchange)
  (maximality := by
    /-
      α : Type u_1
      E : Set α
      Base : Set α → Prop
      exists_finite_base : Exists fun B => And (Base B) B.Finite
      base_exchange : Matroid.ExchangeProperty Base
      subset_ground : ∀ (B : Set α), Base B → HasSubset.Subset B E
      ⊢ ∀ (X : Set α), HasSubset.Subset X E → Matroid.ExistsMaximalSubsetProperty (f …
    -/
    obtain ⟨B, hB, hfin⟩ := exists_finite_base
    refine fun X _ ↦ Matroid.existsMaximalSubsetProperty_of_bdd
      ⟨B.ncard, fun Y ⟨B', hB', hYB'⟩ ↦ ?_⟩ X
    /-
      case intro.intro
      α : Type u_1
      E : Set α
      Base : Set α → Prop
      base_exchange : Matroid.ExchangeProperty Base
      subset_ground : ∀ (B : Set α), Base B → HasSubset.Subset B E
      B : Set α
      hB : Base B
      hfin : B.Finite
      X : Set α
      x✝¹ : HasSubset.Subset X E
      Y : Set α
      x✝ : Exists fun B => And (Base B) (HasSubset.Subset Y B)
      B' : Set α
      hB' : Base B'
      hYB' : HasSubset.Subset Y B'
      ⊢ LE.le Y.encard ↑B.ncard
    -/
    rw [hfin.cast_ncard_eq, base_exchange.encard_base_eq hB hB']
    /-
      case intro.intro
      α : Type u_1
      E : Set α
      Base : Set α → Prop
      base_exchange : Matroid.ExchangeProperty Base
      subset_ground : ∀ (B : Set α), Base B → HasSubset.Subset B E
      B : Set α
      hB : Base B
      hfin : B.Finite
      X : Set α
      x✝¹ : HasSubset.Subset X E
      Y : Set α
      x✝ : Exists fun B => And (Base B) (HasSubset.Subset Y B)
      B' : Set α
      hB' : Base B'
      hYB' : HasSubset.Subset Y B'
      ⊢ LE.le Y.encard B'.encard
    -/
    exact encard_mono hYB')
    /-
      🎉 no goals
    -/
  (subset_ground := subset_ground)


@[simp] theorem ofExistsFiniteBase_base (E : Set α) Base exists_finite_base
    base_exchange subset_ground : (Matroid.ofExistsFiniteBase
      E Base exists_finite_base base_exchange subset_ground).Base = Base := rfl


instance ofExistsFiniteBase_finiteRk (E : Set α) Base exists_finite_base
    base_exchange subset_ground : FiniteRk (Matroid.ofExistsFiniteBase
      E Base exists_finite_base base_exchange subset_ground) := by
  /-
    α : Type u_1
    E : Set α
    Base : Set α → Prop
    exists_finite_base : Exists fun B => And (Base B) B.Finite
    base_exchange : Matroid.ExchangeProperty Base
    subset_ground : ∀ (B : Set α), Base B → HasSubset.Subset B E
    ⊢ (Matroid.ofExistsFiniteBase E Base exists_finite_base base_exchange subset_g …
  -/
  obtain ⟨B, hB, hfin⟩ := exists_finite_base
  /-
    case intro.intro
    α : Type u_1
    E : Set α
    Base : Set α → Prop
    base_exchange : Matroid.ExchangeProperty Base
    subset_ground : ∀ (B : Set α), Base B → HasSubset.Subset B E
    B : Set α
    hB : Base B
    hfin : B.Finite
    ⊢ (Matroid.ofExistsFiniteBase E Base ⋯ base_exchange subset_ground).FiniteRk
  -/
  exact Matroid.Base.finiteRk_of_finite (by simpa) hfin
  /-
    🎉 no goals
  -/


/-- If `E` is finite, then any nonempty collection of its subsets
  with the exchange property is the collection of bases of a matroid on `E`. -/
protected def ofBaseOfFinite {E : Set α} (hE : E.Finite) (Base : Set α → Prop)
    (exists_base : ∃ B, Base B) (base_exchange : ExchangeProperty Base)
    (subset_ground : ∀ B, Base B → B ⊆ E) : Matroid α :=
  Matroid.ofExistsFiniteBase (E := E) (Base := Base)
    (exists_finite_base :=
      let ⟨B, hB⟩ := exists_base
      ⟨B, hB, hE.subset (subset_ground B hB)⟩)
    (base_exchange := base_exchange)
    (subset_ground := subset_ground)


@[simp] theorem ofBaseOfFinite_E {E : Set α} (hE : E.Finite) Base exists_base base_exchange
    subset_ground : (Matroid.ofBaseOfFinite
      hE Base exists_base base_exchange subset_ground).E = E := rfl


@[simp] theorem ofBaseOfFinite_base {E : Set α} (hE : E.Finite) Base exists_base
    base_exchange subset_ground : (Matroid.ofBaseOfFinite
      hE Base exists_base base_exchange subset_ground).Base = Base := rfl


instance ofBaseOfFinite_finite {E : Set α} (hE : E.Finite) Base exists_base
    base_exchange subset_ground : (Matroid.ofBaseOfFinite
      hE Base exists_base base_exchange subset_ground).Finite :=
  ⟨hE⟩


