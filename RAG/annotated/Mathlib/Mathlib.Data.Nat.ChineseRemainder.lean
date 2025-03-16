lemma modEq_list_prod_iff {a b} {l : List ℕ} (co : l.Pairwise Coprime) :
    a ≡ b [MOD l.prod] ↔ ∀ i, a ≡ b [MOD l.get i] := by
  /-
    a b : Nat
    l : List Nat
    co : List.Pairwise Nat.Coprime l
    ⊢ Iff (l.prod.ModEq a b) (∀ (i : Fin l.length), (l.get i).ModEq a b)
  -/
  induction' l with m l ih
    /-
      case nil
      a b : Nat
      co : List.Pairwise Nat.Coprime List.nil
      ⊢ Iff (List.nil.prod.ModEq a b) (∀ (i : Fin List.nil.length), (List.nil.get i) …
    -/
  · simp [modEq_one]
    /-
      🎉 no goals
    -/
    /-
      case cons
      a b m : Nat
      l : List Nat
      ih : List.Pairwise Nat.Coprime l → Iff (l.prod.ModEq a b) (∀ (i : Fin l.length …
      co : List.Pairwise Nat.Coprime (List.cons m l)
      ⊢ Iff ((List.cons m l).prod.ModEq a b) (∀ (i : Fin (List.cons m l).length), (( …
    -/
  · have : Coprime m l.prod := coprime_list_prod_right_iff.mpr (List.pairwise_cons.mp co).1
    simp only [List.prod_cons, ← modEq_and_modEq_iff_modEq_mul this, ih (List.Pairwise.of_cons co),
      List.length_cons]
    /-
      case cons
      a b m : Nat
      l : List Nat
      ih : List.Pairwise Nat.Coprime l → Iff (l.prod.ModEq a b) (∀ (i : Fin l.length …
      co : List.Pairwise Nat.Coprime (List.cons m l)
      this : m.Coprime l.prod
      ⊢ Iff (And (m.ModEq a b) (∀ (i : Fin l.length), (l.get i).ModEq a b)) (∀ (i :  …
    -/
    constructor
      /-
        case cons.mp
        a b m : Nat
        l : List Nat
        ih : List.Pairwise Nat.Coprime l → Iff (l.prod.ModEq a b) (∀ (i : Fin l.length …
        co : List.Pairwise Nat.Coprime (List.cons m l)
        this : m.Coprime l.prod
        ⊢ And (m.ModEq a b) (∀ (i : Fin l.length), (l.get i).ModEq a b) → ∀ (i : Fin ( …
      -/
    · rintro ⟨h0, hs⟩ i
      /-
        case cons.mp.intro
        a b m : Nat
        l : List Nat
        ih : List.Pairwise Nat.Coprime l → Iff (l.prod.ModEq a b) (∀ (i : Fin l.length …
        co : List.Pairwise Nat.Coprime (List.cons m l)
        this : m.Coprime l.prod
        h0 : m.ModEq a b
        hs : ∀ (i : Fin l.length), (l.get i).ModEq a b
        i : Fin (HAdd.hAdd l.length 1)
        ⊢ ((List.cons m l).get i).ModEq a b
      -/
                                  /-
                                    🎉 no goals
                                  -/
      cases i using Fin.cases <;> simp_all
                                  /-
                                    🎉 no goals
                                  -/
      /-
        case cons.mpr
        a b m : Nat
        l : List Nat
        ih : List.Pairwise Nat.Coprime l → Iff (l.prod.ModEq a b) (∀ (i : Fin l.length …
        co : List.Pairwise Nat.Coprime (List.cons m l)
        this : m.Coprime l.prod
        ⊢ (∀ (i : Fin (HAdd.hAdd l.length 1)), ((List.cons m l).get i).ModEq a b) → An …
      -/
    · intro h; exact ⟨h 0, fun i => h i.succ⟩
               /-
                 🎉 no goals
               -/


lemma modEq_list_prod_iff' {a b} {s : ι → ℕ} {l : List ι} (co : l.Pairwise (Coprime on s)) :
    a ≡ b [MOD (l.map s).prod] ↔ ∀ i ∈ l, a ≡ b [MOD s i] := by
  /-
    ι : Type u_1
    a b : Nat
    s : ι → Nat
    l : List ι
    co : List.Pairwise (Function.onFun Nat.Coprime s) l
    ⊢ Iff ((List.map s l).prod.ModEq a b) (∀ (i : ι), Membership.mem l i → (s i).M …
  -/
  induction' l with i l ih
    /-
      case nil
      ι : Type u_1
      a b : Nat
      s : ι → Nat
      co : List.Pairwise (Function.onFun Nat.Coprime s) List.nil
      ⊢ Iff ((List.map s List.nil).prod.ModEq a b) (∀ (i : ι), Membership.mem List.n …
    -/
  · simp [modEq_one]
    /-
      🎉 no goals
    -/
  · have : Coprime (s i) (l.map s).prod := by
      simp only [coprime_list_prod_right_iff, List.mem_map, forall_exists_index, and_imp,
        forall_apply_eq_imp_iff₂]
      intro j hj
      exact (List.pairwise_cons.mp co).1 j hj
    /-
      case cons
      ι : Type u_1
      a b : Nat
      s : ι → Nat
      i : ι
      l : List ι
      ih : List.Pairwise (Function.onFun Nat.Coprime s) l → Iff ((List.map s l).prod …
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ⊢ Iff ((List.map s (List.cons i l)).prod.ModEq a b) (∀ (i_1 : ι), Membership.m …
    -/
    simp [← modEq_and_modEq_iff_modEq_mul this, ih (List.Pairwise.of_cons co)]
    /-
      🎉 no goals
    -/


/-- The natural number less than `(l.map s).prod` congruent to
`a i` mod `s i` for all  `i ∈ l`. -/
def chineseRemainderOfList : (l : List ι) → l.Pairwise (Coprime on s) →
    { k // ∀ i ∈ l, k ≡ a i [MOD s i] }
                         /-
                           ι : Type u_1
                           a s : ι → Nat
                           x✝ : List.Pairwise (Function.onFun Nat.Coprime s) List.nil
                           ⊢ ∀ (i : ι), Membership.mem List.nil i → (s i).ModEq 0 (a i)
                         -/
  | [],     _  => ⟨0, by simp⟩
                         /-
                           🎉 no goals
                         -/
  | i :: l, co => by
    have : Coprime (s i) (l.map s).prod := by
      simp only [coprime_list_prod_right_iff, List.mem_map, forall_exists_index, and_imp,
        forall_apply_eq_imp_iff₂]
      intro j hj
      exact (List.pairwise_cons.mp co).1 j hj
    /-
      ι : Type u_1
      a s : ι → Nat
      i : ι
      l : List ι
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ⊢ Subtype fun k => ∀ (i_1 : ι), Membership.mem (List.cons i l) i_1 → (s i_1).M …
    -/
    have ih := chineseRemainderOfList l co.of_cons
    /-
      ι : Type u_1
      a s : ι → Nat
      i : ι
      l : List ι
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ih : Subtype fun k => ∀ (i : ι), Membership.mem l i → (s i).ModEq k (a i)
      ⊢ Subtype fun k => ∀ (i_1 : ι), Membership.mem (List.cons i l) i_1 → (s i_1).M …
    -/
    have k := chineseRemainder this (a i) ih
    /-
      ι : Type u_1
      a s : ι → Nat
      i : ι
      l : List ι
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ih : Subtype fun k => ∀ (i : ι), Membership.mem l i → (s i).ModEq k (a i)
      k : Subtype fun k => And ((s i).ModEq k (a i)) ((List.map s l).prod.ModEq k ↑ih)
      ⊢ Subtype fun k => ∀ (i_1 : ι), Membership.mem (List.cons i l) i_1 → (s i_1).M …
    -/
    use k
    /-
      case property
      ι : Type u_1
      a s : ι → Nat
      i : ι
      l : List ι
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ih : Subtype fun k => ∀ (i : ι), Membership.mem l i → (s i).ModEq k (a i)
      k : Subtype fun k => And ((s i).ModEq k (a i)) ((List.map s l).prod.ModEq k ↑ih)
      ⊢ ∀ (i_1 : ι), Membership.mem (List.cons i l) i_1 → (s i_1).ModEq (↑k) (a i_1)
    -/
    simp only [List.mem_cons, forall_eq_or_imp, k.prop.1, true_and]
    /-
      case property
      ι : Type u_1
      a s : ι → Nat
      i : ι
      l : List ι
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ih : Subtype fun k => ∀ (i : ι), Membership.mem l i → (s i).ModEq k (a i)
      k : Subtype fun k => And ((s i).ModEq k (a i)) ((List.map s l).prod.ModEq k ↑ih)
      ⊢ ∀ (a_1 : ι), Membership.mem l a_1 → (s a_1).ModEq (↑k) (a a_1)
    -/
    intro j hj
    /-
      case property
      ι : Type u_1
      a s : ι → Nat
      i : ι
      l : List ι
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      this : (s i).Coprime (List.map s l).prod
      ih : Subtype fun k => ∀ (i : ι), Membership.mem l i → (s i).ModEq k (a i)
      k : Subtype fun k => And ((s i).ModEq k (a i)) ((List.map s l).prod.ModEq k ↑ih)
      j : ι
      hj : Membership.mem l j
      ⊢ (s j).ModEq (↑k) (a j)
    -/
    exact ((modEq_list_prod_iff' co.of_cons).mp k.prop.2 j hj).trans (ih.prop j hj)
    /-
      🎉 no goals
    -/


@[simp] theorem chineseRemainderOfList_nil :
    (chineseRemainderOfList a s [] List.Pairwise.nil : ℕ) = 0 := rfl


theorem chineseRemainderOfList_lt_prod (l : List ι)
    (co : l.Pairwise (Coprime on s)) (hs : ∀ i ∈ l, s i ≠ 0) :
    chineseRemainderOfList a s l co < (l.map s).prod := by
  cases l with
  | nil => simp
  | cons i l =>
    simp only [chineseRemainderOfList, List.map_cons, List.prod_cons]
    have : Coprime (s i) (l.map s).prod := by
      simp only [coprime_list_prod_right_iff, List.mem_map, forall_exists_index, and_imp,
        forall_apply_eq_imp_iff₂]
      intro j hj
      exact (List.pairwise_cons.mp co).1 j hj
    refine chineseRemainder_lt_mul this (a i) (chineseRemainderOfList a s l co.of_cons)
      (hs i (List.mem_cons_self _ l)) ?_
    simp only [ne_eq, List.prod_eq_zero_iff, List.mem_map, not_exists, not_and]
    intro j hj
    exact hs j (List.mem_cons_of_mem _ hj)


theorem chineseRemainderOfList_modEq_unique (l : List ι)
    (co : l.Pairwise (Coprime on s)) {z} (hz : ∀ i ∈ l, z ≡ a i [MOD s i]) :
    z ≡ chineseRemainderOfList a s l co [MOD (l.map s).prod] := by
  /-
    ι : Type u_1
    a s : ι → Nat
    l : List ι
    co : List.Pairwise (Function.onFun Nat.Coprime s) l
    z : Nat
    hz : ∀ (i : ι), Membership.mem l i → (s i).ModEq z (a i)
    ⊢ (List.map s l).prod.ModEq z ↑(Nat.chineseRemainderOfList a s l co)
  -/
  induction' l with i l ih
    /-
      case nil
      ι : Type u_1
      a s : ι → Nat
      z : Nat
      co : List.Pairwise (Function.onFun Nat.Coprime s) List.nil
      hz : ∀ (i : ι), Membership.mem List.nil i → (s i).ModEq z (a i)
      ⊢ (List.map s List.nil).prod.ModEq z ↑(Nat.chineseRemainderOfList a s List.nil …
    -/
  · simp [modEq_one]
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      a s : ι → Nat
      z : Nat
      i : ι
      l : List ι
      ih : ∀ (co : List.Pairwise (Function.onFun Nat.Coprime s) l), (∀ (i : ι), Memb …
      co : List.Pairwise (Function.onFun Nat.Coprime s) (List.cons i l)
      hz : ∀ (i_1 : ι), Membership.mem (List.cons i l) i_1 → (s i_1).ModEq z (a i_1)
      ⊢ (List.map s (List.cons i l)).prod.ModEq z ↑(Nat.chineseRemainderOfList a s ( …
    -/
  · simp only [List.map_cons, List.prod_cons, chineseRemainderOfList]
    have : Coprime (s i) (l.map s).prod := by
      simp only [coprime_list_prod_right_iff, List.mem_map, forall_exists_index, and_imp,
        forall_apply_eq_imp_iff₂]
      intro j hj
      exact (List.pairwise_cons.mp co).1 j hj
    exact chineseRemainder_modEq_unique this
      (hz i (List.mem_cons_self _ _)) (ih co.of_cons (fun j hj => hz j (List.mem_cons_of_mem _ hj)))


theorem chineseRemainderOfList_perm {l l' : List ι} (hl : l.Perm l')
    (hs : ∀ i ∈ l, s i ≠ 0) (co : l.Pairwise (Coprime on s)) :
    (chineseRemainderOfList a s l co : ℕ) =
    chineseRemainderOfList a s l' (co.perm hl coprime_comm.mpr) := by
  /-
    ι : Type u_1
    a s : ι → Nat
    l l' : List ι
    hl : l.Perm l'
    hs : ∀ (i : ι), Membership.mem l i → Ne (s i) 0
    co : List.Pairwise (Function.onFun Nat.Coprime s) l
    ⊢ Eq ↑(Nat.chineseRemainderOfList a s l co) ↑(Nat.chineseRemainderOfList a s l …
  -/
  let z := chineseRemainderOfList a s l' (co.perm hl coprime_comm.mpr)
  /-
    ι : Type u_1
    a s : ι → Nat
    l l' : List ι
    hl : l.Perm l'
    hs : ∀ (i : ι), Membership.mem l i → Ne (s i) 0
    co : List.Pairwise (Function.onFun Nat.Coprime s) l
    z : Subtype fun k => ∀ (i : ι), Membership.mem l' i → (s i).ModEq k (a i) := N …
    ⊢ Eq ↑(Nat.chineseRemainderOfList a s l co) ↑(Nat.chineseRemainderOfList a s l …
  -/
  have hlp : (l.map s).prod = (l'.map s).prod := List.Perm.prod_eq (List.Perm.map s hl)
  exact (chineseRemainderOfList_modEq_unique a s l co (z := z)
    (fun i hi => z.prop i (hl.symm.mem_iff.mpr hi))).symm.eq_of_lt_of_lt
      (chineseRemainderOfList_lt_prod _ _ _ _ hs)
      (by rw [hlp]
          exact chineseRemainderOfList_lt_prod _ _ _ _
            (by simpa [List.Perm.mem_iff hl.symm] using hs))


/-- The natural number less than `(m.map s).prod` congruent to
`a i` mod `s i` for all  `i ∈ m`. -/
def chineseRemainderOfMultiset {m : Multiset ι} :
    m.Nodup → (∀ i ∈ m, s i ≠ 0) → Set.Pairwise {x | x ∈ m} (Coprime on s) →
    { k // ∀ i ∈ m, k ≡ a i [MOD s i] } :=
  Quotient.recOn m
    (fun l nod _ co =>
      chineseRemainderOfList a s l (List.Nodup.pairwise_of_forall_ne nod co))
    (fun l l' (pp : l.Perm l') ↦
      funext fun nod' : l'.Nodup =>
      have nod : l.Nodup := pp.symm.nodup_iff.mp nod'
      funext fun hs' : ∀ i ∈ l', s i ≠ 0 =>
                                        /-
                                          ι : Type u_1
                                          a s : ι → Nat
                                          m : Multiset ι
                                          l l' : List ι
                                          pp : l.Perm l'
                                          nod' : l'.Nodup
                                          nod : l.Nodup
                                          hs' : ∀ (i : ι), Membership.mem l' i → Ne (s i) 0
                                          ⊢ ∀ (i : ι), Membership.mem l i → Ne (s i) 0
                                        -/
      have hs : ∀ i ∈ l, s i ≠ 0  := by simpa [List.Perm.mem_iff pp] using hs'
                                        /-
                                          🎉 no goals
                                        -/
      funext fun co' : Set.Pairwise {x | x ∈ l'} (Coprime on s) =>
                                                              /-
                                                                ι : Type u_1
                                                                a s : ι → Nat
                                                                m : Multiset ι
                                                                l l' : List ι
                                                                pp : l.Perm l'
                                                                nod' : l'.Nodup
                                                                nod : l.Nodup
                                                                hs' : ∀ (i : ι), Membership.mem l' i → Ne (s i) 0
                                                                hs : ∀ (i : ι), Membership.mem l i → Ne (s i) 0
                                                                co' : (setOf fun x => Membership.mem l' x).Pairwise (Function.onFun Nat.Coprim …
                                                                ⊢ (setOf fun x => Membership.mem l x).Pairwise (Function.onFun Nat.Coprime s)
                                                              -/
      have co : Set.Pairwise {x | x ∈ l} (Coprime on s) := by simpa [List.Perm.mem_iff pp] using co'
                                                              /-
                                                                🎉 no goals
                                                              -/
      have lco : l.Pairwise (Coprime on s) := List.Nodup.pairwise_of_forall_ne nod co
      have : ∀ {m' e nod'' hs'' co''}, @Eq.ndrec (Multiset ι) l
        (fun m ↦ m.Nodup → (∀ i ∈ m, s i ≠ 0) →
          Set.Pairwise {x | x ∈ m} (Coprime on s) → { k // ∀ i ∈ m, k ≡ a i [MOD s i] })
        (fun nod _ co ↦ chineseRemainderOfList a s l (List.Nodup.pairwise_of_forall_ne nod co))
          m' e nod'' hs'' co'' =
        (chineseRemainderOfList a s l lco : ℕ) := by
          /-
            ι : Type u_1
            a s : ι → Nat
            m : Multiset ι
            l l' : List ι
            pp : l.Perm l'
            nod' : l'.Nodup
            nod : l.Nodup
            hs' : ∀ (i : ι), Membership.mem l' i → Ne (s i) 0
            hs : ∀ (i : ι), Membership.mem l i → Ne (s i) 0
            co' : (setOf fun x => Membership.mem l' x).Pairwise (Function.onFun Nat.Coprim …
            co : (setOf fun x => Membership.mem l x).Pairwise (Function.onFun Nat.Coprime s)
            lco : List.Pairwise (Function.onFun Nat.Coprime s) l
            ⊢ ∀ {m' : Multiset ι} {e : Eq (↑l) m'} {nod'' : m'.Nodup} {hs'' : ∀ (i : ι), M …
          -/
          rintro _ rfl _ _ _; rfl
                              /-
                                🎉 no goals
                              -/
         /-
           ι : Type u_1
           a s : ι → Nat
           m : Multiset ι
           l l' : List ι
           pp : l.Perm l'
           nod' : l'.Nodup
           nod : l.Nodup
           hs' : ∀ (i : ι), Membership.mem l' i → Ne (s i) 0
           hs : ∀ (i : ι), Membership.mem l i → Ne (s i) 0
           co' : (setOf fun x => Membership.mem l' x).Pairwise (Function.onFun Nat.Coprim …
           co : (setOf fun x => Membership.mem l x).Pairwise (Function.onFun Nat.Coprime s)
           lco : List.Pairwise (Function.onFun Nat.Coprime s) l
           this : ∀ {m' : Multiset ι} {e : Eq (↑l) m'} {nod'' : m'.Nodup} {hs'' : ∀ (i :  …
           ⊢ Eq (Eq.ndrec (motive := fun x => x.Nodup → (∀ (i : ι), Membership.mem x i →  …
         -/
      by ext; exact this.trans <| chineseRemainderOfList_perm a s pp hs lco)
              /-
                🎉 no goals
              -/


theorem chineseRemainderOfMultiset_lt_prod {m : Multiset ι}
    (nod : m.Nodup) (hs : ∀ i ∈ m, s i ≠ 0) (pp : Set.Pairwise {x | x ∈ m} (Coprime on s)) :
    chineseRemainderOfMultiset a s nod hs pp < (m.map s).prod := by
  /-
    ι : Type u_1
    a s : ι → Nat
    m : Multiset ι
    nod : m.Nodup
    hs : ∀ (i : ι), Membership.mem m i → Ne (s i) 0
    pp : (setOf fun x => Membership.mem m x).Pairwise (Function.onFun Nat.Coprime s)
    ⊢ LT.lt (↑(Nat.chineseRemainderOfMultiset a s nod hs pp)) (Multiset.map s m).p …
  -/
  induction' m using Quot.ind with l
  /-
    case mk
    ι : Type u_1
    a s : ι → Nat
    l : List ι
    nod : Multiset.Nodup (Quot.mk (⇑(List.isSetoid ι)) l)
    hs : ∀ (i : ι), Membership.mem (Quot.mk (⇑(List.isSetoid ι)) l) i → Ne (s i) 0
    pp : (setOf fun x => Membership.mem (Quot.mk (⇑(List.isSetoid ι)) l) x).Pairwi …
    ⊢ LT.lt (↑(Nat.chineseRemainderOfMultiset a s nod hs pp)) (Multiset.map s (Quo …
  -/
  unfold chineseRemainderOfMultiset
  simpa using chineseRemainderOfList_lt_prod a s l
    (List.Nodup.pairwise_of_forall_ne nod pp) (by simpa using hs)


/-- The natural number less than `∏ i ∈ t, s i` congruent to
`a i` mod `s i` for all  `i ∈ t`. -/
def chineseRemainderOfFinset (t : Finset ι)
    (hs : ∀ i ∈ t, s i ≠ 0) (pp : Set.Pairwise t (Coprime on s)) :
    { k // ∀ i ∈ t, k ≡ a i [MOD s i] } := by
  /-
    ι : Type u_1
    a s : ι → Nat
    t : Finset ι
    hs : ∀ (i : ι), Membership.mem t i → Ne (s i) 0
    pp : (↑t).Pairwise (Function.onFun Nat.Coprime s)
    ⊢ Subtype fun k => ∀ (i : ι), Membership.mem t i → (s i).ModEq k (a i)
  -/
  simpa using chineseRemainderOfMultiset a s t.nodup (by simpa using hs) (by simpa using pp)
  /-
    🎉 no goals
  -/


theorem chineseRemainderOfFinset_lt_prod {t : Finset ι}
    (hs : ∀ i ∈ t, s i ≠ 0) (pp : Set.Pairwise t (Coprime on s)) :
    chineseRemainderOfFinset a s t hs pp < ∏ i ∈ t, s i := by
  simpa [chineseRemainderOfFinset] using
    chineseRemainderOfMultiset_lt_prod a s t.nodup (by simpa using hs) (by simpa using pp)


