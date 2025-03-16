/-- The lower boundary of an embedding `e : Embedding c c'`, as a predicate on `ι`.
It is satisfied by `j : ι` when there exists `i' : ι'` not in the image of `e.f`
such that `c'.Rel i' (e.f j)`. -/
def BoundaryGE (j : ι) : Prop :=
  c'.Rel (c'.prev (e.f j)) (e.f j) ∧ ∀ i, ¬c'.Rel (e.f i) (e.f j)


lemma boundaryGE {i' : ι'} {j : ι} (hj : c'.Rel i' (e.f j)) (hi' : ∀ i, e.f i ≠ i') :
    e.BoundaryGE j := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    i' : ι'
    j : ι
    hj : c'.Rel i' (e.f j)
    hi' : ∀ (i : ι), Ne (e.f i) i'
    ⊢ e.BoundaryGE j
  -/
  constructor
    /-
      case left
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i' : ι'
      j : ι
      hj : c'.Rel i' (e.f j)
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ c'.Rel (c'.prev (e.f j)) (e.f j)
    -/
  · simpa only [c'.prev_eq' hj] using hj
    /-
      🎉 no goals
    -/
    /-
      case right
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i' : ι'
      j : ι
      hj : c'.Rel i' (e.f j)
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ ∀ (i : ι), Not (c'.Rel (e.f i) (e.f j))
    -/
  · intro i hi
    /-
      case right
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i' : ι'
      j : ι
      hj : c'.Rel i' (e.f j)
      hi' : ∀ (i : ι), Ne (e.f i) i'
      i : ι
      hi : c'.Rel (e.f i) (e.f j)
      ⊢ False
    -/
    apply hi' i
    /-
      case right
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i' : ι'
      j : ι
      hj : c'.Rel i' (e.f j)
      hi' : ∀ (i : ι), Ne (e.f i) i'
      i : ι
      hi : c'.Rel (e.f i) (e.f j)
      ⊢ Eq (e.f i) i'
    -/
    rw [← c'.prev_eq' hj, c'.prev_eq' hi]
    /-
      🎉 no goals
    -/


lemma not_boundaryGE_next [e.IsRelIff] {j k : ι} (hk : c.Rel j k) :
    ¬ e.BoundaryGE k := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    j k : ι
    hk : c.Rel j k
    ⊢ Not (e.BoundaryGE k)
  -/
  dsimp [BoundaryGE]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    j k : ι
    hk : c.Rel j k
    ⊢ Not (And (c'.Rel (c'.prev (e.f k)) (e.f k)) (∀ (i : ι), Not (c'.Rel (e.f i)  …
  -/
  simp only [not_and, not_forall, not_not]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    j k : ι
    hk : c.Rel j k
    ⊢ c'.Rel (c'.prev (e.f k)) (e.f k) → Exists fun x => c'.Rel (e.f x) (e.f k)
  -/
  intro
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    j k : ι
    hk : c.Rel j k
    a✝ : c'.Rel (c'.prev (e.f k)) (e.f k)
    ⊢ Exists fun x => c'.Rel (e.f x) (e.f k)
  -/
  exact ⟨j, by simpa only [e.rel_iff] using hk⟩
  /-
    🎉 no goals
  -/


lemma not_boundaryGE_next' [e.IsRelIff] {j k : ι} (hj : ¬ e.BoundaryGE j) (hk : c.next j = k) :
    ¬ e.BoundaryGE k := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    j k : ι
    hj : Not (e.BoundaryGE j)
    hk : Eq (c.next j) k
    ⊢ Not (e.BoundaryGE k)
  -/
  by_cases hjk : c.Rel j k
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j k : ι
      hj : Not (e.BoundaryGE j)
      hk : Eq (c.next j) k
      hjk : c.Rel j k
      ⊢ Not (e.BoundaryGE k)
    -/
  · exact e.not_boundaryGE_next hjk
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j k : ι
      hj : Not (e.BoundaryGE j)
      hk : Eq (c.next j) k
      hjk : Not (c.Rel j k)
      ⊢ Not (e.BoundaryGE k)
    -/
  · subst hk
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryGE j)
      hjk : Not (c.Rel j (c.next j))
      ⊢ Not (e.BoundaryGE (c.next j))
    -/
    simpa only [c.next_eq_self j hjk] using hj
    /-
      🎉 no goals
    -/


variable {e} in
lemma BoundaryGE.not_mem {j : ι} (hj : e.BoundaryGE j) {i' : ι'} (hi' : c'.Rel i' (e.f j))
    (a : ι) : e.f a ≠ i' := fun ha =>
             /-
               ι : Type u_1
               ι' : Type u_2
               c : ComplexShape ι
               c' : ComplexShape ι'
               e : c.Embedding c'
               j : ι
               hj : e.BoundaryGE j
               i' : ι'
               hi' : c'.Rel i' (e.f j)
               a : ι
               ha : Eq (e.f a) i'
               ⊢ c'.Rel (e.f a) (e.f j)
             -/
  hj.2 a (by simpa only [ha] using hi')
             /-
               🎉 no goals
             -/


lemma prev_f_of_not_boundaryGE [e.IsRelIff] {i j : ι} (hij : c.prev j = i)
    (hj : ¬ e.BoundaryGE j) :
    c'.prev (e.f j) = e.f i := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    hij : Eq (c.prev j) i
    hj : Not (e.BoundaryGE j)
    ⊢ Eq (c'.prev (e.f j)) (e.f i)
  -/
  by_cases hij' : c.Rel i j
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      i j : ι
      hij : Eq (c.prev j) i
      hj : Not (e.BoundaryGE j)
      hij' : c.Rel i j
      ⊢ Eq (c'.prev (e.f j)) (e.f i)
    -/
  · exact c'.prev_eq' (by simpa only [e.rel_iff] using hij')
    /-
      🎉 no goals
    -/
  · obtain rfl : j = i := by
      simpa only [c.prev_eq_self j (by simpa only [hij] using hij')] using hij
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryGE j)
      hij : Eq (c.prev j) j
      hij' : Not (c.Rel j j)
      ⊢ Eq (c'.prev (e.f j)) (e.f j)
    -/
    apply c'.prev_eq_self
    /-
      case neg.hj
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryGE j)
      hij : Eq (c.prev j) j
      hij' : Not (c.Rel j j)
      ⊢ Not (c'.Rel (c'.prev (e.f j)) (e.f j))
    -/
    intro hj'
    /-
      case neg.hj
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryGE j)
      hij : Eq (c.prev j) j
      hij' : Not (c.Rel j j)
      hj' : c'.Rel (c'.prev (e.f j)) (e.f j)
      ⊢ False
    -/
    simp only [BoundaryGE, not_and, not_forall, not_not] at hj
    /-
      case neg.hj
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hij : Eq (c.prev j) j
      hij' : Not (c.Rel j j)
      hj' : c'.Rel (c'.prev (e.f j)) (e.f j)
      hj : c'.Rel (c'.prev (e.f j)) (e.f j) → Exists fun x => c'.Rel (e.f x) (e.f j)
      ⊢ False
    -/
    obtain ⟨i, hi⟩ := hj hj'
    /-
      case neg.hj.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hij : Eq (c.prev j) j
      hij' : Not (c.Rel j j)
      hj' : c'.Rel (c'.prev (e.f j)) (e.f j)
      hj : c'.Rel (c'.prev (e.f j)) (e.f j) → Exists fun x => c'.Rel (e.f x) (e.f j)
      i : ι
      hi : c'.Rel (e.f i) (e.f j)
      ⊢ False
    -/
    rw [e.rel_iff] at hi
    /-
      case neg.hj.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hij : Eq (c.prev j) j
      hij' : Not (c.Rel j j)
      hj' : c'.Rel (c'.prev (e.f j)) (e.f j)
      hj : c'.Rel (c'.prev (e.f j)) (e.f j) → Exists fun x => c'.Rel (e.f x) (e.f j)
      i : ι
      hi : c.Rel i j
      ⊢ False
    -/
    rw [c.prev_eq' hi] at hij
    /-
      case neg.hj.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hij' : Not (c.Rel j j)
      hj' : c'.Rel (c'.prev (e.f j)) (e.f j)
      hj : c'.Rel (c'.prev (e.f j)) (e.f j) → Exists fun x => c'.Rel (e.f x) (e.f j)
      i : ι
      hij : Eq i j
      hi : c.Rel i j
      ⊢ False
    -/
    exact hij' (by simpa only [hij] using hi)
    /-
      🎉 no goals
    -/


variable {e} in
lemma BoundaryGE.false_of_isTruncLE {j : ι} (hj : e.BoundaryGE j) [e.IsTruncLE] : False := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    j : ι
    hj : e.BoundaryGE j
    inst✝ : e.IsTruncLE
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := e.mem_prev hj.1
  /-
    case intro
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    j : ι
    hj : e.BoundaryGE j
    inst✝ : e.IsTruncLE
    i : ι
    hi : Eq (e.f i) (c'.prev (e.f j))
    ⊢ False
  -/
  exact hj.2 i (by simpa only [hi] using hj.1)
  /-
    🎉 no goals
  -/


/-- The upper boundary of an embedding `e : Embedding c c'`, as a predicate on `ι`.
It is satisfied by `j : ι` when there exists `k' : ι'` not in the image of `e.f`
such that `c'.Rel (e.f j) k'`. -/
def BoundaryLE (j : ι) : Prop :=
  c'.Rel (e.f j) (c'.next (e.f j)) ∧ ∀ k, ¬c'.Rel (e.f j) (e.f k)


lemma boundaryLE {k' : ι'} {j : ι} (hj : c'.Rel (e.f j) k') (hk' : ∀ i, e.f i ≠ k') :
    e.BoundaryLE j := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    k' : ι'
    j : ι
    hj : c'.Rel (e.f j) k'
    hk' : ∀ (i : ι), Ne (e.f i) k'
    ⊢ e.BoundaryLE j
  -/
  constructor
    /-
      case left
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      k' : ι'
      j : ι
      hj : c'.Rel (e.f j) k'
      hk' : ∀ (i : ι), Ne (e.f i) k'
      ⊢ c'.Rel (e.f j) (c'.next (e.f j))
    -/
  · simpa only [c'.next_eq' hj] using hj
    /-
      🎉 no goals
    -/
    /-
      case right
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      k' : ι'
      j : ι
      hj : c'.Rel (e.f j) k'
      hk' : ∀ (i : ι), Ne (e.f i) k'
      ⊢ ∀ (k : ι), Not (c'.Rel (e.f j) (e.f k))
    -/
  · intro k hk
    /-
      case right
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      k' : ι'
      j : ι
      hj : c'.Rel (e.f j) k'
      hk' : ∀ (i : ι), Ne (e.f i) k'
      k : ι
      hk : c'.Rel (e.f j) (e.f k)
      ⊢ False
    -/
    apply hk' k
    /-
      case right
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      k' : ι'
      j : ι
      hj : c'.Rel (e.f j) k'
      hk' : ∀ (i : ι), Ne (e.f i) k'
      k : ι
      hk : c'.Rel (e.f j) (e.f k)
      ⊢ Eq (e.f k) k'
    -/
    rw [← c'.next_eq' hj, c'.next_eq' hk]
    /-
      🎉 no goals
    -/


lemma not_boundaryLE_prev [e.IsRelIff] {i j : ι} (hi : c.Rel i j) :
    ¬ e.BoundaryLE i := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    hi : c.Rel i j
    ⊢ Not (e.BoundaryLE i)
  -/
  dsimp [BoundaryLE]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    hi : c.Rel i j
    ⊢ Not (And (c'.Rel (e.f i) (c'.next (e.f i))) (∀ (k : ι), Not (c'.Rel (e.f i)  …
  -/
  simp only [not_and, not_forall, not_not]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    hi : c.Rel i j
    ⊢ c'.Rel (e.f i) (c'.next (e.f i)) → Exists fun x => c'.Rel (e.f i) (e.f x)
  -/
  intro
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    hi : c.Rel i j
    a✝ : c'.Rel (e.f i) (c'.next (e.f i))
    ⊢ Exists fun x => c'.Rel (e.f i) (e.f x)
  -/
  exact ⟨j, by simpa only [e.rel_iff] using hi⟩
  /-
    🎉 no goals
  -/


lemma not_boundaryLE_prev' [e.IsRelIff] {i j : ι} (hj : ¬ e.BoundaryLE j) (hk : c.prev j = i) :
    ¬ e.BoundaryLE i := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    hj : Not (e.BoundaryLE j)
    hk : Eq (c.prev j) i
    ⊢ Not (e.BoundaryLE i)
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      i j : ι
      hj : Not (e.BoundaryLE j)
      hk : Eq (c.prev j) i
      hij : c.Rel i j
      ⊢ Not (e.BoundaryLE i)
    -/
  · exact e.not_boundaryLE_prev hij
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      i j : ι
      hj : Not (e.BoundaryLE j)
      hk : Eq (c.prev j) i
      hij : Not (c.Rel i j)
      ⊢ Not (e.BoundaryLE i)
    -/
  · subst hk
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryLE j)
      hij : Not (c.Rel (c.prev j) j)
      ⊢ Not (e.BoundaryLE (c.prev j))
    -/
    simpa only [c.prev_eq_self j hij] using hj
    /-
      🎉 no goals
    -/


variable {e} in
lemma BoundaryLE.not_mem {j : ι} (hj : e.BoundaryLE j) {k' : ι'} (hk' : c'.Rel (e.f j) k')
    (a : ι) : e.f a ≠ k' := fun ha =>
             /-
               ι : Type u_1
               ι' : Type u_2
               c : ComplexShape ι
               c' : ComplexShape ι'
               e : c.Embedding c'
               j : ι
               hj : e.BoundaryLE j
               k' : ι'
               hk' : c'.Rel (e.f j) k'
               a : ι
               ha : Eq (e.f a) k'
               ⊢ c'.Rel (e.f j) (e.f a)
             -/
  hj.2 a (by simpa only [ha] using hk')
             /-
               🎉 no goals
             -/


lemma next_f_of_not_boundaryLE [e.IsRelIff] {j k : ι} (hjk : c.next j = k)
    (hj : ¬ e.BoundaryLE j) :
    c'.next (e.f j) = e.f k := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    j k : ι
    hjk : Eq (c.next j) k
    hj : Not (e.BoundaryLE j)
    ⊢ Eq (c'.next (e.f j)) (e.f k)
  -/
  by_cases hjk' : c.Rel j k
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j k : ι
      hjk : Eq (c.next j) k
      hj : Not (e.BoundaryLE j)
      hjk' : c.Rel j k
      ⊢ Eq (c'.next (e.f j)) (e.f k)
    -/
  · exact c'.next_eq' (by simpa only [e.rel_iff] using hjk')
    /-
      🎉 no goals
    -/
  · obtain rfl : j = k := by
      simpa only [c.next_eq_self j (by simpa only [hjk] using hjk')] using hjk
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryLE j)
      hjk : Eq (c.next j) j
      hjk' : Not (c.Rel j j)
      ⊢ Eq (c'.next (e.f j)) (e.f j)
    -/
    apply c'.next_eq_self
    /-
      case neg.hj
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryLE j)
      hjk : Eq (c.next j) j
      hjk' : Not (c.Rel j j)
      ⊢ Not (c'.Rel (e.f j) (c'.next (e.f j)))
    -/
    intro hj'
    /-
      case neg.hj
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hj : Not (e.BoundaryLE j)
      hjk : Eq (c.next j) j
      hjk' : Not (c.Rel j j)
      hj' : c'.Rel (e.f j) (c'.next (e.f j))
      ⊢ False
    -/
    simp only [BoundaryLE, not_and, not_forall, not_not] at hj
    /-
      case neg.hj
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hjk : Eq (c.next j) j
      hjk' : Not (c.Rel j j)
      hj' : c'.Rel (e.f j) (c'.next (e.f j))
      hj : c'.Rel (e.f j) (c'.next (e.f j)) → Exists fun x => c'.Rel (e.f j) (e.f x)
      ⊢ False
    -/
    obtain ⟨k, hk⟩ := hj hj'
    /-
      case neg.hj.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hjk : Eq (c.next j) j
      hjk' : Not (c.Rel j j)
      hj' : c'.Rel (e.f j) (c'.next (e.f j))
      hj : c'.Rel (e.f j) (c'.next (e.f j)) → Exists fun x => c'.Rel (e.f j) (e.f x)
      k : ι
      hk : c'.Rel (e.f j) (e.f k)
      ⊢ False
    -/
    rw [e.rel_iff] at hk
    /-
      case neg.hj.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hjk : Eq (c.next j) j
      hjk' : Not (c.Rel j j)
      hj' : c'.Rel (e.f j) (c'.next (e.f j))
      hj : c'.Rel (e.f j) (c'.next (e.f j)) → Exists fun x => c'.Rel (e.f j) (e.f x)
      k : ι
      hk : c.Rel j k
      ⊢ False
    -/
    rw [c.next_eq' hk] at hjk
    /-
      case neg.hj.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      j : ι
      hjk' : Not (c.Rel j j)
      hj' : c'.Rel (e.f j) (c'.next (e.f j))
      hj : c'.Rel (e.f j) (c'.next (e.f j)) → Exists fun x => c'.Rel (e.f j) (e.f x)
      k : ι
      hjk : Eq k j
      hk : c.Rel j k
      ⊢ False
    -/
    exact hjk' (by simpa only [hjk] using hk)
    /-
      🎉 no goals
    -/


variable {e} in
lemma BoundaryLE.false_of_isTruncGE {j : ι} (hj : e.BoundaryLE j) [e.IsTruncGE] : False := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    j : ι
    hj : e.BoundaryLE j
    inst✝ : e.IsTruncGE
    ⊢ False
  -/
  obtain ⟨k, hk⟩ := e.mem_next hj.1
  /-
    case intro
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    j : ι
    hj : e.BoundaryLE j
    inst✝ : e.IsTruncGE
    k : ι
    hk : Eq (e.f k) (c'.next (e.f j))
    ⊢ False
  -/
  exact hj.2 k (by simpa only [hk] using hj.1)
  /-
    🎉 no goals
  -/


lemma boundaryGE_embeddingUpIntGE_iff (p : ℤ) (n : ℕ) :
    (embeddingUpIntGE p).BoundaryGE n ↔ n = 0 := by
  /-
    p : Int
    n : Nat
    ⊢ Iff ((ComplexShape.embeddingUpIntGE p).BoundaryGE n) (Eq n 0)
  -/
  constructor
    /-
      case mp
      p : Int
      n : Nat
      ⊢ (ComplexShape.embeddingUpIntGE p).BoundaryGE n → Eq n 0
    -/
  · intro h
    /-
      case mp
      p : Int
      n : Nat
      h : (ComplexShape.embeddingUpIntGE p).BoundaryGE n
      ⊢ Eq n 0
    -/
    obtain _|n := n
      /-
        case mp.zero
        p : Int
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE 0
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.succ
        p : Int
        n : Nat
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE (HAdd.hAdd n 1)
        ⊢ Eq (HAdd.hAdd n 1) 0
      -/
    · have := h.2 n
      /-
        case mp.succ
        p : Int
        n : Nat
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE (HAdd.hAdd n 1)
        this : Not ((ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE p).f n)  …
        ⊢ Eq (HAdd.hAdd n 1) 0
      -/
      dsimp at this
      /-
        case mp.succ
        p : Int
        n : Nat
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE (HAdd.hAdd n 1)
        this : Not (Eq (HAdd.hAdd (HAdd.hAdd p ↑n) 1) (HAdd.hAdd p ↑(HAdd.hAdd n 1)))
        ⊢ Eq (HAdd.hAdd n 1) 0
      -/
      omega
      /-
        🎉 no goals
      -/
    /-
      case mpr
      p : Int
      n : Nat
      ⊢ Eq n 0 → (ComplexShape.embeddingUpIntGE p).BoundaryGE n
    -/
  · rintro rfl
    /-
      case mpr
      p : Int
      ⊢ (ComplexShape.embeddingUpIntGE p).BoundaryGE 0
    -/
    constructor
      /-
        case mpr.left
        p : Int
        ⊢ (ComplexShape.up Int).Rel ((ComplexShape.up Int).prev ((ComplexShape.embeddi …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mpr.right
        p : Int
        ⊢ ∀ (i : Nat), Not ((ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE  …
      -/
    · intro i hi
      /-
        case mpr.right
        p : Int
        i : Nat
        hi : (ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE p).f i) ((Compl …
        ⊢ False
      -/
      dsimp at hi
      /-
        case mpr.right
        p : Int
        i : Nat
        hi : Eq (HAdd.hAdd (HAdd.hAdd p ↑i) 1) (HAdd.hAdd p 0)
        ⊢ False
      -/
      omega
      /-
        🎉 no goals
      -/


lemma boundaryLE_embeddingUpIntLE_iff (p : ℤ) (n : ℕ) :
    (embeddingUpIntGE p).BoundaryGE n ↔ n = 0 := by
  /-
    p : Int
    n : Nat
    ⊢ Iff ((ComplexShape.embeddingUpIntGE p).BoundaryGE n) (Eq n 0)
  -/
  constructor
    /-
      case mp
      p : Int
      n : Nat
      ⊢ (ComplexShape.embeddingUpIntGE p).BoundaryGE n → Eq n 0
    -/
  · intro h
    /-
      case mp
      p : Int
      n : Nat
      h : (ComplexShape.embeddingUpIntGE p).BoundaryGE n
      ⊢ Eq n 0
    -/
    obtain _|n := n
      /-
        case mp.zero
        p : Int
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE 0
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.succ
        p : Int
        n : Nat
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE (HAdd.hAdd n 1)
        ⊢ Eq (HAdd.hAdd n 1) 0
      -/
    · have := h.2 n
      /-
        case mp.succ
        p : Int
        n : Nat
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE (HAdd.hAdd n 1)
        this : Not ((ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE p).f n)  …
        ⊢ Eq (HAdd.hAdd n 1) 0
      -/
      dsimp at this
      /-
        case mp.succ
        p : Int
        n : Nat
        h : (ComplexShape.embeddingUpIntGE p).BoundaryGE (HAdd.hAdd n 1)
        this : Not (Eq (HAdd.hAdd (HAdd.hAdd p ↑n) 1) (HAdd.hAdd p ↑(HAdd.hAdd n 1)))
        ⊢ Eq (HAdd.hAdd n 1) 0
      -/
      omega
      /-
        🎉 no goals
      -/
    /-
      case mpr
      p : Int
      n : Nat
      ⊢ Eq n 0 → (ComplexShape.embeddingUpIntGE p).BoundaryGE n
    -/
  · rintro rfl
    /-
      case mpr
      p : Int
      ⊢ (ComplexShape.embeddingUpIntGE p).BoundaryGE 0
    -/
    constructor
      /-
        case mpr.left
        p : Int
        ⊢ (ComplexShape.up Int).Rel ((ComplexShape.up Int).prev ((ComplexShape.embeddi …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mpr.right
        p : Int
        ⊢ ∀ (i : Nat), Not ((ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE  …
      -/
    · intro i hi
      /-
        case mpr.right
        p : Int
        i : Nat
        hi : (ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE p).f i) ((Compl …
        ⊢ False
      -/
      dsimp at hi
      /-
        case mpr.right
        p : Int
        i : Nat
        hi : Eq (HAdd.hAdd (HAdd.hAdd p ↑i) 1) (HAdd.hAdd p 0)
        ⊢ False
      -/
      omega
      /-
        🎉 no goals
      -/


