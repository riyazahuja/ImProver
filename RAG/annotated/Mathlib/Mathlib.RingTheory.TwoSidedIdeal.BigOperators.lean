lemma listSum_mem {ι : Type*} (l : List ι) (f : ι → R) (hl : ∀ x ∈ l, f x ∈ I) :
    (l.map f).sum ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    l : List ι
    f : ι → R
    hl : ∀ (x : ι), Membership.mem l x → Membership.mem I (f x)
    ⊢ Membership.mem I (List.map f l).sum
  -/
  rw [mem_iff, ← List.sum_map_zero]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    l : List ι
    f : ι → R
    hl : ∀ (x : ι), Membership.mem l x → Membership.mem I (f x)
    ⊢ I.ringCon (List.map f l).sum (List.map (fun x => 0) ?m.353).sum
  -/
  exact I.ringCon.listSum l hl
  /-
    🎉 no goals
  -/


lemma multisetSum_mem {ι : Type*} (s : Multiset ι) (f : ι → R) (hs : ∀ x ∈ s, f x ∈ I) :
    (s.map f).sum ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Multiset ι
    f : ι → R
    hs : ∀ (x : ι), Membership.mem s x → Membership.mem I (f x)
    ⊢ Membership.mem I (Multiset.map f s).sum
  -/
  rw [mem_iff, ← Multiset.sum_map_zero]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Multiset ι
    f : ι → R
    hs : ∀ (x : ι), Membership.mem s x → Membership.mem I (f x)
    ⊢ I.ringCon (Multiset.map f s).sum (Multiset.map (fun x => 0) ?m.1290).sum
  -/
  exact I.ringCon.multisetSum s hs
  /-
    🎉 no goals
  -/


lemma finsetSum_mem {ι : Type*} (s : Finset ι) (f : ι → R) (hs : ∀ x ∈ s, f x ∈ I) :
    s.sum f ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Finset ι
    f : ι → R
    hs : ∀ (x : ι), Membership.mem s x → Membership.mem I (f x)
    ⊢ Membership.mem I (s.sum f)
  -/
  rw [mem_iff, ← Finset.sum_const_zero]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Finset ι
    f : ι → R
    hs : ∀ (x : ι), Membership.mem s x → Membership.mem I (f x)
    ⊢ I.ringCon (s.sum f) (Finset.sum ?m.1847 fun _x => 0)
  -/
  exact I.ringCon.finsetSum s hs
  /-
    🎉 no goals
  -/


lemma listProd_mem {ι : Type*} (l : List ι) (f : ι → R) (hl : ∃ x ∈ l, f x ∈ I) :
    (l.map f).prod ∈ I := by
  induction l with
  | nil => simp only [List.not_mem_nil, false_and, exists_false] at hl
  | cons x l ih =>
    simp only [List.mem_cons, exists_eq_or_imp] at hl
    rcases hl with h | hal
    · simpa only [List.map_cons, List.prod_cons] using I.mul_mem_right _ _ h
    · simpa only [List.map_cons, List.prod_cons] using I.mul_mem_left _ _ <| ih hal


lemma multiSetProd_mem {ι : Type*} (s : Multiset ι) (f : ι → R) (hs : ∃ x ∈ s, f x ∈ I) :
    (s.map f).prod ∈ I := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Multiset ι
    f : ι → R
    hs : Exists fun x => And (Membership.mem s x) (Membership.mem I (f x))
    ⊢ Membership.mem I (Multiset.map f s).prod
  -/
  rcases s
  /-
    case mk
    R : Type u_1
    inst✝ : CommRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Multiset ι
    f : ι → R
    a✝ : List ι
    hs : Exists fun x => And (Membership.mem (Quot.mk (⇑(List.isSetoid ι)) a✝) x)  …
    ⊢ Membership.mem I (Multiset.map f (Quot.mk (⇑(List.isSetoid ι)) a✝)).prod
  -/
  simpa using listProd_mem (hl := hs)
  /-
    🎉 no goals
  -/


lemma finsetProd_mem {ι : Type*} (s : Finset ι) (f : ι → R) (hs : ∃ x ∈ s, f x ∈ I) :
    s.prod f ∈ I := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    s : Finset ι
    f : ι → R
    hs : Exists fun x => And (Membership.mem s x) (Membership.mem I (f x))
    ⊢ Membership.mem I (s.prod f)
  -/
  rcases s
  /-
    case mk
    R : Type u_1
    inst✝ : CommRing R
    I : TwoSidedIdeal R
    ι : Type u_2
    f : ι → R
    val✝ : Multiset ι
    nodup✝ : val✝.Nodup
    hs : Exists fun x => And (Membership.mem { val := val✝, nodup := nodup✝ } x) ( …
    ⊢ Membership.mem I ({ val := val✝, nodup := nodup✝ }.prod f)
  -/
  simpa using multiSetProd_mem (hs := hs)
  /-
    🎉 no goals
  -/


