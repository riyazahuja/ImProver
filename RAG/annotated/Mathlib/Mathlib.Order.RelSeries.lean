/--
Let `r` be a relation on `α`, a relation series of `r` of length `n` is a series
`a_0, a_1, ..., a_n` such that `r a_i a_{i+1}` for all `i < n`
-/
structure RelSeries where
  /-- The number of inequalities in the series -/
  length : ℕ
  /-- The underlying function of a relation series -/
  toFun : Fin (length + 1) → α
  /-- Adjacent elements are related -/
  step : ∀ (i : Fin length), r (toFun (Fin.castSucc i)) (toFun i.succ)


instance : CoeFun (RelSeries r) (fun x ↦ Fin (x.length + 1) → α) :=
{ coe := RelSeries.toFun }


/--
For any type `α`, each term of `α` gives a relation series with the right most index to be 0.
-/
@[simps!] def singleton (a : α) : RelSeries r where
  length := 0
  toFun _ := a
  step := Fin.elim0


instance [IsEmpty α] : IsEmpty (RelSeries r) where
  false x := IsEmpty.false (x 0)


instance [Inhabited α] : Inhabited (RelSeries r) where
  default := singleton r default


instance [Nonempty α] : Nonempty (RelSeries r) :=
  Nonempty.map (singleton r) inferInstance


@[ext (iff := false)]
lemma ext {x y : RelSeries r} (length_eq : x.length = y.length)
                                                 /-
                                                   α : Type u_1
                                                   r : Rel α α
                                                   β : Type u_2
                                                   s : Rel β β
                                                   x y : RelSeries r
                                                   length_eq : Eq x.length y.length
                                                   ⊢ Eq (HAdd.hAdd x.length 1) (HAdd.hAdd y.length 1)
                                                 -/
    (toFun_eq : x.toFun = y.toFun ∘ Fin.cast (by rw [length_eq])) : x = y := by
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    α : Type u_1
    r : Rel α α
    x y : RelSeries r
    length_eq : Eq x.length y.length
    toFun_eq : Eq x.toFun (Function.comp y.toFun (Fin.cast ⋯))
    ⊢ Eq x y
  -/
  rcases x with ⟨nx, fx⟩
  /-
    case mk
    α : Type u_1
    r : Rel α α
    y : RelSeries r
    nx : Nat
    fx : Fin (HAdd.hAdd nx 1) → α
    step✝ : ∀ (i : Fin nx), r (fx i.castSucc) (fx i.succ)
    length_eq : Eq { length := nx, toFun := fx, step := step✝ }.length y.length
    toFun_eq : Eq { length := nx, toFun := fx, step := step✝ }.toFun (Function.com …
    ⊢ Eq { length := nx, toFun := fx, step := step✝ } y
  -/
  dsimp only at length_eq toFun_eq
  /-
    case mk
    α : Type u_1
    r : Rel α α
    y : RelSeries r
    nx : Nat
    fx : Fin (HAdd.hAdd nx 1) → α
    step✝ : ∀ (i : Fin nx), r (fx i.castSucc) (fx i.succ)
    length_eq : Eq nx y.length
    toFun_eq : Eq fx (Function.comp y.toFun (Fin.cast ⋯))
    ⊢ Eq { length := nx, toFun := fx, step := step✝ } y
  -/
  subst length_eq toFun_eq
  /-
    case mk
    α : Type u_1
    r : Rel α α
    y : RelSeries r
    step✝ : ∀ (i : Fin y.length), r (Function.comp y.toFun (Fin.cast ⋯) i.castSucc …
    ⊢ Eq { length := y.length, toFun := Function.comp y.toFun (Fin.cast ⋯), step : …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma rel_of_lt [IsTrans α r] (x : RelSeries r) {i j : Fin (x.length + 1)} (h : i < j) :
    r (x i) (x j) :=
  (Fin.liftFun_iff_succ r).mpr x.step h


lemma rel_or_eq_of_le [IsTrans α r] (x : RelSeries r) {i j : Fin (x.length + 1)} (h : i ≤ j) :
    r (x i) (x j) ∨ x i = x j :=
  (Fin.lt_or_eq_of_le h).imp (x.rel_of_lt ·) (by rw [·])


/--
Given two relations `r, s` on `α` such that `r ≤ s`, any relation series of `r` induces a relation
series of `s`
-/
@[simps!]
def ofLE (x : RelSeries r) {s : Rel α α} (h : r ≤ s) : RelSeries s where
  length := x.length
  toFun := x
  step _ := h _ _ <| x.step _


lemma coe_ofLE (x : RelSeries r) {s : Rel α α} (h : r ≤ s) :
    (x.ofLE h : _ → _) = x := rfl


/-- Every relation series gives a list -/
def toList (x : RelSeries r) : List α := List.ofFn x


@[simp]
lemma length_toList (x : RelSeries r) : x.toList.length = x.length + 1 :=
  List.length_ofFn _


lemma toList_chain' (x : RelSeries r) : x.toList.Chain' r := by
  /-
    α : Type u_1
    r : Rel α α
    x : RelSeries r
    ⊢ List.Chain' r x.toList
  -/
  rw [List.chain'_iff_get]
  /-
    α : Type u_1
    r : Rel α α
    x : RelSeries r
    ⊢ ∀ (i : Nat) (h : LT.lt i (HSub.hSub x.toList.length 1)), r (x.toList.get ⟨i, …
  -/
  intros i h
  /-
    α : Type u_1
    r : Rel α α
    x : RelSeries r
    i : Nat
    h : LT.lt i (HSub.hSub x.toList.length 1)
    ⊢ r (x.toList.get ⟨i, ⋯⟩) (x.toList.get ⟨HAdd.hAdd i 1, ⋯⟩)
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  convert x.step ⟨i, by simpa [toList] using h⟩ <;> apply List.get_ofFn
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma toList_ne_nil (x : RelSeries r) : x.toList ≠ [] := fun m =>
  List.eq_nil_iff_forall_not_mem.mp m (x 0) <| (List.mem_ofFn _ _).mpr ⟨_, rfl⟩


/-- Every nonempty list satisfying the chain condition gives a relation series -/
@[simps]
def fromListChain' (x : List α) (x_ne_nil : x ≠ []) (hx : x.Chain' r) : RelSeries r where
  length := x.length - 1
  toFun i := x[Fin.cast (Nat.succ_pred_eq_of_pos <| List.length_pos.mpr x_ne_nil) i]
  step i := List.chain'_iff_get.mp hx i i.2


/-- Relation series of `r` and nonempty list of `α` satisfying `r`-chain condition bijectively
corresponds to each other. -/
protected def Equiv : RelSeries r ≃ {x : List α | x ≠ [] ∧ x.Chain' r} where
  toFun x := ⟨_, x.toList_ne_nil, x.toList_chain'⟩
  invFun x := fromListChain' _ x.2.1 x.2.2
                        /-
                          α : Type u_1
                          r : Rel α α
                          β : Type u_2
                          s : Rel β β
                          x : RelSeries r
                          ⊢ Eq ((fun x => RelSeries.fromListChain' ↑x ⋯ ⋯) ((fun x => ⟨x.toList, ⋯⟩) x)) …
                        -/
                        /-
                          🎉 no goals
                        -/
  left_inv x := ext (by simp [toList]) <| by ext; dsimp; apply List.get_ofFn
                                                         /-
                                                           🎉 no goals
                                                         -/
  right_inv x := by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      x : ↑(setOf fun x => And (Ne x List.nil) (List.Chain' r x))
      ⊢ Eq ((fun x => ⟨x.toList, ⋯⟩) ((fun x => RelSeries.fromListChain' ↑x ⋯ ⋯) x)) x
    -/
    refine Subtype.ext (List.ext_get ?_ fun n hn1 _ => by dsimp; apply List.get_ofFn)
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      x : ↑(setOf fun x => And (Ne x List.nil) (List.Chain' r x))
      ⊢ Eq (↑((fun x => ⟨x.toList, ⋯⟩) ((fun x => RelSeries.fromListChain' ↑x ⋯ ⋯) x …
    -/
    have := Nat.succ_pred_eq_of_pos <| List.length_pos.mpr x.2.1
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      x : ↑(setOf fun x => And (Ne x List.nil) (List.Chain' r x))
      this : Eq (↑x).length.pred.succ (↑x).length
      ⊢ Eq (↑((fun x => ⟨x.toList, ⋯⟩) ((fun x => RelSeries.fromListChain' ↑x ⋯ ⋯) x …
    -/
    simp_all [toList]
    /-
      🎉 no goals
    -/


lemma toList_injective : Function.Injective (RelSeries.toList (r := r)) :=
  fun _ _ h ↦ (RelSeries.Equiv).injective <| Subtype.ext h

-- TODO : build a similar bijection between `RelSeries α` and `Quiver.Path`


/-- A relation `r` is said to be finite dimensional iff there is a relation series of `r` with the
  maximum length. -/
@[mk_iff]
class FiniteDimensional : Prop where
  /-- A relation `r` is said to be finite dimensional iff there is a relation series of `r` with the
    maximum length. -/
  exists_longest_relSeries : ∃ x : RelSeries r, ∀ y : RelSeries r, y.length ≤ x.length


/-- A relation `r` is said to be infinite dimensional iff there exists relation series of arbitrary
  length. -/
@[mk_iff]
class InfiniteDimensional : Prop where
  /-- A relation `r` is said to be infinite dimensional iff there exists relation series of
    arbitrary length. -/
  exists_relSeries_with_length : ∀ n : ℕ, ∃ x : RelSeries r, x.length = n


/-- The longest relational series when a relation is finite dimensional -/
protected noncomputable def longestOf [r.FiniteDimensional] : RelSeries r :=
  Rel.FiniteDimensional.exists_longest_relSeries.choose


lemma length_le_length_longestOf [r.FiniteDimensional] (x : RelSeries r) :
    x.length ≤ (RelSeries.longestOf r).length :=
  Rel.FiniteDimensional.exists_longest_relSeries.choose_spec _


/-- A relation series with length `n` if the relation is infinite dimensional -/
protected noncomputable def withLength [r.InfiniteDimensional] (n : ℕ) : RelSeries r :=
  (Rel.InfiniteDimensional.exists_relSeries_with_length n).choose


@[simp] lemma length_withLength [r.InfiniteDimensional] (n : ℕ) :
    (RelSeries.withLength r n).length = n :=
  (Rel.InfiniteDimensional.exists_relSeries_with_length n).choose_spec


/-- If a relation on `α` is infinite dimensional, then `α` is nonempty. -/
lemma nonempty_of_infiniteDimensional [r.InfiniteDimensional] : Nonempty α :=
  ⟨RelSeries.withLength r 0 0⟩


instance membership : Membership α (RelSeries r) :=
  ⟨Function.swap (· ∈ Set.range ·)⟩


theorem mem_def : x ∈ s ↔ x ∈ Set.range s := Iff.rfl


@[simp] theorem mem_toList : x ∈ s.toList ↔ x ∈ s := by
  /-
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    x : α
    ⊢ Iff (Membership.mem s.toList x) (Membership.mem s x)
  -/
  rw [RelSeries.toList, List.mem_ofFn, RelSeries.mem_def]
  /-
    🎉 no goals
  -/


theorem subsingleton_of_length_eq_zero (hs : s.length = 0) : {x | x ∈ s}.Subsingleton := by
  /-
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    hs : Eq s.length 0
    ⊢ (setOf fun x => Membership.mem s x).Subsingleton
  -/
  rintro - ⟨i, rfl⟩ - ⟨j, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    hs : Eq s.length 0
    i j : Fin (HAdd.hAdd s.length 1)
    ⊢ Eq (s.toFun i) (s.toFun j)
  -/
  congr!
  /-
    case intro.intro.h.e'_4
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    hs : Eq s.length 0
    i j : Fin (HAdd.hAdd s.length 1)
    ⊢ Eq i j
  -/
  exact finCongr (by rw [hs, zero_add]) |>.injective <| Subsingleton.elim (α := Fin 1) _ _
  /-
    🎉 no goals
  -/


theorem length_ne_zero_of_nontrivial (h : {x | x ∈ s}.Nontrivial) : s.length ≠ 0 :=
  fun hs ↦ h.not_subsingleton <| subsingleton_of_length_eq_zero hs


theorem length_pos_of_nontrivial (h : {x | x ∈ s}.Nontrivial) : 0 < s.length :=
  Nat.pos_iff_ne_zero.mpr <| length_ne_zero_of_nontrivial h


theorem length_ne_zero (irrefl : Irreflexive r) : s.length ≠ 0 ↔ {x | x ∈ s}.Nontrivial := by
  refine ⟨fun h ↦ ⟨s 0, by simp [mem_def], s 1, by simp [mem_def], fun rid ↦ irrefl (s 0) ?_⟩,
    length_ne_zero_of_nontrivial⟩
  /-
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    irrefl : Irreflexive r
    h : Ne s.length 0
    rid : Eq (s.toFun 0) (s.toFun 1)
    ⊢ r (s.toFun 0) (s.toFun 0)
  -/
  nth_rw 2 [rid]
  /-
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    irrefl : Irreflexive r
    h : Ne s.length 0
    rid : Eq (s.toFun 0) (s.toFun 1)
    ⊢ r (s.toFun 0) (s.toFun 1)
  -/
  convert s.step ⟨0, by omega⟩
  /-
    case h.e'_2.h.e'_4
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    irrefl : Irreflexive r
    h : Ne s.length 0
    rid : Eq (s.toFun 0) (s.toFun 1)
    ⊢ Eq 1 ⟨0, ⋯⟩.succ
  -/
  ext
  /-
    case h.e'_2.h.e'_4.h
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    irrefl : Irreflexive r
    h : Ne s.length 0
    rid : Eq (s.toFun 0) (s.toFun 1)
    ⊢ Eq ↑1 ↑⟨0, ⋯⟩.succ
  -/
  simpa [Nat.pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem length_pos (irrefl : Irreflexive r) : 0 < s.length ↔ {x | x ∈ s}.Nontrivial :=
  Nat.pos_iff_ne_zero.trans <| length_ne_zero irrefl


lemma length_eq_zero (irrefl : Irreflexive r) : s.length = 0 ↔ {x | x ∈ s}.Subsingleton := by
  /-
    α : Type u_1
    r : Rel α α
    s : RelSeries r
    irrefl : Irreflexive r
    ⊢ Iff (Eq s.length 0) (setOf fun x => Membership.mem s x).Subsingleton
  -/
  rw [← not_ne_iff, length_ne_zero irrefl, Set.not_nontrivial_iff]
  /-
    🎉 no goals
  -/


/-- Start of a series, i.e. for `a₀ -r→ a₁ -r→ ... -r→ aₙ`, its head is `a₀`.

Since a relation series is assumed to be non-empty, this is well defined. -/
def head (x : RelSeries r) : α := x 0


/-- End of a series, i.e. for `a₀ -r→ a₁ -r→ ... -r→ aₙ`, its last element is `aₙ`.

Since a relation series is assumed to be non-empty, this is well defined. -/
def last (x : RelSeries r) : α := x <| Fin.last _


lemma apply_last (x : RelSeries r) : x (Fin.last <| x.length) = x.last := rfl


lemma head_mem (x : RelSeries r) : x.head ∈ x := ⟨_, rfl⟩


lemma last_mem (x : RelSeries r) : x.last ∈ x := ⟨_, rfl⟩


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              r : Rel α α
                                                                              x : α
                                                                              ⊢ Eq (RelSeries.singleton r x).head x
                                                                            -/
lemma head_singleton {r : Rel α α} (x : α) : (singleton r x).head = x := by simp [singleton, head]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              r : Rel α α
                                                                              x : α
                                                                              ⊢ Eq (RelSeries.singleton r x).last x
                                                                            -/
lemma last_singleton {r : Rel α α} (x : α) : (singleton r x).last = x := by simp [singleton, last]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/--
If `a₀ -r→ a₁ -r→ ... -r→ aₙ` and `b₀ -r→ b₁ -r→ ... -r→ bₘ` are two strict series
such that `r aₙ b₀`, then there is a chain of length `n + m + 1` given by
`a₀ -r→ a₁ -r→ ... -r→ aₙ -r→ b₀ -r→ b₁ -r→ ... -r→ bₘ`.
-/
@[simps length]
def append (p q : RelSeries r) (connect : r p.last q.head) : RelSeries r where
  length := p.length + q.length + 1
                                         /-
                                           α : Type u_1
                                           r : Rel α α
                                           β : Type u_2
                                           s : Rel β β
                                           p q : RelSeries r
                                           connect : r p.last q.head
                                           ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd p.length q.length) 1) 1) (HAdd.hAdd (HAd …
                                         -/
  toFun := Fin.append p q ∘ Fin.cast (by omega)
                                         /-
                                           🎉 no goals
                                         -/
  step i := by
    obtain hi | rfl | hi :=
      lt_trichotomy i (Fin.castLE (by omega) (Fin.last _ : Fin (p.length + 1)))
      /-
        case inl
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
        hi : LT.lt i (Fin.castLE ⋯ (Fin.last p.length))
        ⊢ r (Function.comp (Fin.append p.toFun q.toFun) (Fin.cast ⋯) i.castSucc) (Func …
      -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    · convert p.step ⟨i.1, hi⟩ <;> convert Fin.append_left p q _ <;> rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
      /-
        case inr.inl
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        ⊢ r (Function.comp (Fin.append p.toFun q.toFun) (Fin.cast ⋯) (Fin.castLE ⋯ (Fi …
      -/
    · convert connect
        /-
          case h.e'_1
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : r p.last q.head
          ⊢ Eq (Function.comp (Fin.append p.toFun q.toFun) (Fin.cast ⋯) (Fin.castLE ⋯ (F …
        -/
      · convert Fin.append_left p q _
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : r p.last q.head
          ⊢ Eq (Function.comp (Fin.append p.toFun q.toFun) (Fin.cast ⋯) (Fin.castLE ⋯ (F …
        -/
      · convert Fin.append_right p q _; rfl
                                        /-
                                          🎉 no goals
                                        -/
      /-
        case inr.inr
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
        hi : LT.lt (Fin.castLE ⋯ (Fin.last p.length)) i
        ⊢ r (Function.comp (Fin.append p.toFun q.toFun) (Fin.cast ⋯) i.castSucc) (Func …
      -/
    · set x := _; set y := _
      /-
        case inr.inr
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
        hi : LT.lt (Fin.castLE ⋯ (Fin.last p.length)) i
        x : ?m.45461 := ?m.45462
        y : ?m.45667 := ?m.45668
        ⊢ r (Function.comp (Fin.append p.toFun q.toFun) (Fin.cast ⋯) i.castSucc) (Func …
      -/
      change r (Fin.append p q x) (Fin.append p q y)
      have hx : x = Fin.natAdd _ ⟨i - (p.length + 1), Nat.sub_lt_left_of_lt_add hi <|
          i.2.trans <| by omega⟩ := by
        ext; dsimp [x, y]; rw [Nat.add_sub_cancel']; exact hi
      have hy : y = Fin.natAdd _ ⟨i - p.length, Nat.sub_lt_left_of_lt_add (le_of_lt hi)
          (by exact i.2)⟩ := by
        ext
        dsimp
        conv_rhs => rw [Nat.add_comm p.length 1, add_assoc,
          Nat.add_sub_cancel' <| le_of_lt (show p.length < i.1 from hi), add_comm]
        rfl
      /-
        case inr.inr
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
        hi : LT.lt (Fin.castLE ⋯ (Fin.last p.length)) i
        x : Fin (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) := Fin.cast  …
        y : Fin (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) := Fin.cast  …
        hx : Eq x (Fin.natAdd (HAdd.hAdd p.length 1) ⟨HSub.hSub (↑i) (HAdd.hAdd p.leng …
        hy : Eq y (Fin.natAdd (HAdd.hAdd p.length 1) ⟨HSub.hSub (↑i) p.length, ⋯⟩)
        ⊢ r (Fin.append p.toFun q.toFun x) (Fin.append p.toFun q.toFun y)
      -/
      rw [hx, Fin.append_right, hy, Fin.append_right]
      /-
        case inr.inr
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
        hi : LT.lt (Fin.castLE ⋯ (Fin.last p.length)) i
        x : Fin (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) := Fin.cast  …
        y : Fin (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) := Fin.cast  …
        hx : Eq x (Fin.natAdd (HAdd.hAdd p.length 1) ⟨HSub.hSub (↑i) (HAdd.hAdd p.leng …
        hy : Eq y (Fin.natAdd (HAdd.hAdd p.length 1) ⟨HSub.hSub (↑i) p.length, ⋯⟩)
        ⊢ r (q.toFun ⟨HSub.hSub (↑i) (HAdd.hAdd p.length 1), ⋯⟩) (q.toFun ⟨HSub.hSub ( …
      -/
      convert q.step ⟨i - (p.length + 1), Nat.sub_lt_left_of_lt_add hi <| by omega⟩
      rw [Fin.succ_mk, Nat.sub_eq_iff_eq_add (le_of_lt hi : p.length ≤ i),
        Nat.add_assoc _ 1, add_comm 1, Nat.sub_add_cancel]
      /-
        case h.e'_2.h.e'_4.h.e'_2
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : r p.last q.head
        i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
        hi : LT.lt (Fin.castLE ⋯ (Fin.last p.length)) i
        x : Fin (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) := Fin.cast  …
        y : Fin (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) := Fin.cast  …
        hx : Eq x (Fin.natAdd (HAdd.hAdd p.length 1) ⟨HSub.hSub (↑i) (HAdd.hAdd p.leng …
        hy : Eq y (Fin.natAdd (HAdd.hAdd p.length 1) ⟨HSub.hSub (↑i) p.length, ⋯⟩)
        ⊢ LE.le (HAdd.hAdd p.length 1) ↑i
      -/
      exact hi
      /-
        🎉 no goals
      -/


lemma append_apply_left (p q : RelSeries r) (connect : r p.last q.head)
    (i : Fin (p.length + 1)) :
                                                            /-
                                                              α : Type u_1
                                                              r : Rel α α
                                                              β : Type u_2
                                                              s : Rel β β
                                                              p q : RelSeries r
                                                              connect : r p.last q.head
                                                              i : Fin (HAdd.hAdd p.length 1)
                                                              ⊢ Eq (HAdd.hAdd (HAdd.hAdd p.length 1) (HAdd.hAdd q.length 1)) (HAdd.hAdd (p.a …
                                                            -/
    p.append q connect ((i.castAdd (q.length + 1)).cast (by dsimp; omega)) = p i := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq ((p.append q connect).toFun (Fin.cast ⋯ (Fin.castAdd (HAdd.hAdd q.length  …
  -/
  delta append
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq ({ length := HAdd.hAdd (HAdd.hAdd p.length q.length) 1, toFun := Function …
  -/
  simp only [Function.comp_apply]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (Fin.append p.toFun q.toFun (Fin.cast ⋯ (Fin.cast ⋯ (Fin.castAdd (HAdd.hA …
  -/
  convert Fin.append_left _ _ _
  /-
    🎉 no goals
  -/


lemma append_apply_right (p q : RelSeries r) (connect : r p.last q.head)
    (i : Fin (q.length + 1)) :
    p.append q connect (i.natAdd p.length + 1) = q i := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq ((p.append q connect).toFun (HAdd.hAdd (↑↑(Fin.natAdd p.length i)) 1)) (q …
  -/
  delta append
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq ({ length := HAdd.hAdd (HAdd.hAdd p.length q.length) 1, toFun := Function …
  -/
  simp only [Fin.coe_natAdd, Nat.cast_add, Function.comp_apply]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq (Fin.append p.toFun q.toFun (Fin.cast ⋯ (HAdd.hAdd (↑(HAdd.hAdd p.length  …
  -/
  convert Fin.append_right _ _ _
  /-
    case h.e'_2.h.e'_6.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq (Fin.cast ⋯ (HAdd.hAdd (↑(HAdd.hAdd p.length ↑i)) 1)) (Fin.natAdd (HAdd.h …
  -/
  ext
  /-
    case h.e'_2.h.e'_6.h.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq ↑(Fin.cast ⋯ (HAdd.hAdd (↑(HAdd.hAdd p.length ↑i)) 1)) ↑(Fin.natAdd (HAdd …
  -/
  simp only [Fin.coe_cast, Fin.coe_natAdd]
  /-
    case h.e'_2.h.e'_6.h.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq (↑(HAdd.hAdd (↑(HAdd.hAdd p.length ↑i)) 1)) (HAdd.hAdd (HAdd.hAdd p.lengt …
  -/
  conv_rhs => rw [add_assoc, add_comm 1, ← add_assoc]
  /-
    case h.e'_2.h.e'_6.h.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq (↑(HAdd.hAdd (↑(HAdd.hAdd p.length ↑i)) 1)) (HAdd.hAdd (HAdd.hAdd p.lengt …
  -/
  change _ % _ = _
  /-
    case h.e'_2.h.e'_6.h.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ Eq (HMod.hMod (HAdd.hAdd (HMod.hMod (HAdd.hAdd p.length ↑i) (HAdd.hAdd (HAdd …
  -/
  simp only [Nat.add_mod_mod, Nat.mod_add_mod, Nat.one_mod, Nat.mod_succ_eq_iff_lt]
  /-
    case h.e'_2.h.e'_6.h.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    i : Fin (HAdd.hAdd q.length 1)
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd p.length ↑i) 1) (HAdd.hAdd (HAdd.hAdd p.length q …
  -/
  omega
  /-
    🎉 no goals
  -/


@[simp] lemma head_append (p q : RelSeries r) (connect : r p.last q.head) :
    (p.append q connect).head = p.head :=
  append_apply_left p q connect 0


@[simp] lemma last_append (p q : RelSeries r) (connect : r p.last q.head) :
    (p.append q connect).last = q.last := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    ⊢ Eq (p.append q connect).last q.last
  -/
  delta last
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    ⊢ Eq ((p.append q connect).toFun (Fin.last (p.append q connect).length)) (q.to …
  -/
  convert append_apply_right p q connect (Fin.last _)
  /-
    case h.e'_2.h.e'_4
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    ⊢ Eq (Fin.last (p.append q connect).length) (HAdd.hAdd (↑↑(Fin.natAdd p.length …
  -/
  ext
  /-
    case h.e'_2.h.e'_4.h
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : r p.last q.head
    ⊢ Eq ↑(Fin.last (p.append q connect).length) ↑(HAdd.hAdd (↑↑(Fin.natAdd p.leng …
  -/
  change _ = _ % _
  simp only [append_length, Fin.val_last, Fin.natAdd_last, Nat.one_mod, Nat.mod_add_mod,
    Nat.mod_succ]


/--
For two types `α, β` and relation on them `r, s`, if `f : α → β` preserves relation `r`, then an
`r`-series can be pushed out to an `s`-series by
`a₀ -r→ a₁ -r→ ... -r→ aₙ ↦ f a₀ -s→ f a₁ -s→ ... -s→ f aₙ`
-/
@[simps length]
def map (p : RelSeries r) (f : r →r s) : RelSeries s where
  length := p.length
  toFun := f.1.comp p
  step := (f.2 <| p.step ·)


@[simp] lemma map_apply (p : RelSeries r) (f : r →r s) (i : Fin (p.length + 1)) :
    p.map f i = f (p i) := rfl


@[simp] lemma head_map (p : RelSeries r) (f : r →r s) : (p.map f).head = f p.head := rfl


@[simp] lemma last_map (p : RelSeries r) (f : r →r s) : (p.map f).last = f p.last := rfl


/--
If `a₀ -r→ a₁ -r→ ... -r→ aₙ` is an `r`-series and `a` is such that
`aᵢ -r→ a -r→ a_ᵢ₊₁`, then
`a₀ -r→ a₁ -r→ ... -r→ aᵢ -r→ a -r→ aᵢ₊₁ -r→ ... -r→ aₙ`
is another `r`-series
-/
@[simps]
def insertNth (p : RelSeries r) (i : Fin p.length) (a : α)
    (prev_connect : r (p (Fin.castSucc i)) a) (connect_next : r a (p i.succ)) : RelSeries r where
  toFun := (Fin.castSucc i.succ).insertNth a p
  step m := by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin p.length
      a : α
      prev_connect : r (p.toFun i.castSucc) a
      connect_next : r a (p.toFun i.succ)
      m : Fin (HAdd.hAdd p.length 1)
      ⊢ r (i.succ.castSucc.insertNth a p.toFun m.castSucc) (i.succ.castSucc.insertNt …
    -/
    set x := _; set y := _; change r x y
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin p.length
      a : α
      prev_connect : r (p.toFun i.castSucc) a
      connect_next : r a (p.toFun i.succ)
      m : Fin (HAdd.hAdd p.length 1)
      x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
      y : α := i.succ.castSucc.insertNth a p.toFun m.succ
      ⊢ r x y
    -/
    obtain hm | hm | hm := lt_trichotomy m.1 i.1
      /-
        case inl
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        a : α
        prev_connect : r (p.toFun i.castSucc) a
        connect_next : r a (p.toFun i.succ)
        m : Fin (HAdd.hAdd p.length 1)
        x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
        y : α := i.succ.castSucc.insertNth a p.toFun m.succ
        hm : LT.lt ↑m ↑i
        ⊢ r x y
      -/
    · convert p.step ⟨m, hm.trans i.2⟩
        /-
          case h.e'_1
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq x (p.toFun ⟨↑m, ⋯⟩.castSucc)
        -/
      · show Fin.insertNth _ _ _ _ = _
        /-
          case h.e'_1
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq (i.succ.castSucc.insertNth a p.toFun m.castSucc) (p.toFun ⟨↑m, ⋯⟩.castSucc)
        -/
        rw [Fin.insertNth_apply_below]
        /-
          case h.e'_1
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.castSucc.castPred ⋯))) (p.toFun ⟨↑m, ⋯⟩.castSucc)
        -/
        pick_goal 2
          /-
            case h.e'_1.h
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt ↑m ↑i
            ⊢ LT.lt m.castSucc i.succ.castSucc
          -/
        · exact hm.trans (lt_add_one _)
          /-
            🎉 no goals
          -/
        /-
          case h.e'_1
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.castSucc.castPred ⋯))) (p.toFun ⟨↑m, ⋯⟩.castSucc)
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq y (p.toFun ⟨↑m, ⋯⟩.succ)
        -/
      · show Fin.insertNth _ _ _ _ = _
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq (i.succ.castSucc.insertNth a p.toFun m.succ) (p.toFun ⟨↑m, ⋯⟩.succ)
        -/
        rw [Fin.insertNth_apply_below]
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.succ.castPred ⋯))) (p.toFun ⟨↑m, ⋯⟩.succ)
        -/
        pick_goal 2
          /-
            case h.e'_2.h
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt ↑m ↑i
            ⊢ LT.lt m.succ i.succ.castSucc
          -/
        · change m.1 + 1 < i.1 + 1; rwa [add_lt_add_iff_right]
                                    /-
                                      🎉 no goals
                                    -/
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt ↑m ↑i
          ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.succ.castPred ⋯))) (p.toFun ⟨↑m, ⋯⟩.succ)
        -/
        simp; rfl
              /-
                🎉 no goals
              -/
    · rw [show x = p m from show Fin.insertNth _ _ _ _ = _ by
        rw [Fin.insertNth_apply_below]
        pick_goal 2
        · show m.1 < i.1 + 1; exact hm ▸ lt_add_one _
        simp]
      /-
        case inr.inl
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        a : α
        prev_connect : r (p.toFun i.castSucc) a
        connect_next : r a (p.toFun i.succ)
        m : Fin (HAdd.hAdd p.length 1)
        x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
        y : α := i.succ.castSucc.insertNth a p.toFun m.succ
        hm : Eq ↑m ↑i
        ⊢ r (p.toFun m) y
      -/
      convert prev_connect
        /-
          case h.e'_1.h.e'_4
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : Eq ↑m ↑i
          ⊢ Eq m i.castSucc
        -/
      · ext; exact hm
             /-
               🎉 no goals
             -/
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : Eq ↑m ↑i
          ⊢ Eq y a
        -/
      · change Fin.insertNth _ _ _ _ = _
        rw [show m.succ = i.succ.castSucc by ext; change _ + 1 = _ + 1; rw [hm],
          Fin.insertNth_apply_same]
      /-
        case inr.inr
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        a : α
        prev_connect : r (p.toFun i.castSucc) a
        connect_next : r a (p.toFun i.succ)
        m : Fin (HAdd.hAdd p.length 1)
        x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
        y : α := i.succ.castSucc.insertNth a p.toFun m.succ
        hm : LT.lt ↑i ↑m
        ⊢ r x y
      -/
    · rw [Nat.lt_iff_add_one_le, le_iff_lt_or_eq] at hm
      /-
        case inr.inr
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        a : α
        prev_connect : r (p.toFun i.castSucc) a
        connect_next : r a (p.toFun i.succ)
        m : Fin (HAdd.hAdd p.length 1)
        x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
        y : α := i.succ.castSucc.insertNth a p.toFun m.succ
        hm : Or (LT.lt (HAdd.hAdd (↑i) 1) ↑m) (Eq (HAdd.hAdd (↑i) 1) ↑m)
        ⊢ r x y
      -/
      obtain hm | hm := hm
        /-
          case inr.inr.inl
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
          ⊢ r x y
        -/
      · convert p.step ⟨m.1 - 1, Nat.sub_lt_right_of_lt_add (by omega) m.2⟩
          /-
            case h.e'_1
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq x (p.toFun ⟨HSub.hSub (↑m) 1, ⋯⟩.castSucc)
          -/
        · change Fin.insertNth _ _ _ _ = _
          /-
            case h.e'_1
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (i.succ.castSucc.insertNth a p.toFun m.castSucc) (p.toFun ⟨HSub.hSub (↑m) …
          -/
          rw [Fin.insertNth_apply_above (h := hm)]
          /-
            case h.e'_1
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.castSucc.pred ⋯))) (p.toFun ⟨HSub.hSub (↑m) 1, ⋯⟩ …
          -/
          aesop
          /-
            🎉 no goals
          -/
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq y (p.toFun ⟨HSub.hSub (↑m) 1, ⋯⟩.succ)
          -/
        · change Fin.insertNth _ _ _ _ = _
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (i.succ.castSucc.insertNth a p.toFun m.succ) (p.toFun ⟨HSub.hSub (↑m) 1,  …
          -/
          rw [Fin.insertNth_apply_above]
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.succ.pred ⋯))) (p.toFun ⟨HSub.hSub (↑m) 1, ⋯⟩.succ)
          -/
          swap
            /-
              case h.e'_2.h
              α : Type u_1
              r : Rel α α
              β : Type u_2
              s : Rel β β
              p : RelSeries r
              i : Fin p.length
              a : α
              prev_connect : r (p.toFun i.castSucc) a
              connect_next : r a (p.toFun i.succ)
              m : Fin (HAdd.hAdd p.length 1)
              x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
              y : α := i.succ.castSucc.insertNth a p.toFun m.succ
              hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
              ⊢ LT.lt i.succ.castSucc m.succ
            -/
          · exact hm.trans (lt_add_one _)
            /-
              🎉 no goals
            -/
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.succ.pred ⋯))) (p.toFun ⟨HSub.hSub (↑m) 1, ⋯⟩.succ)
          -/
          simp only [Fin.val_succ, Fin.pred_succ, eq_rec_constant, Fin.succ_mk]
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (p.toFun m) (p.toFun ⟨HAdd.hAdd (HSub.hSub (↑m) 1) 1, ⋯⟩)
          -/
          congr
          /-
            case h.e'_2.e_a
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : LT.lt (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq m ⟨HAdd.hAdd (HSub.hSub (↑m) 1) 1, ⋯⟩
          -/
          exact Fin.ext <| Eq.symm <| Nat.succ_pred_eq_of_pos (lt_trans (Nat.zero_lt_succ _) hm)
          /-
            🎉 no goals
          -/
        /-
          case inr.inr.inr
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p : RelSeries r
          i : Fin p.length
          a : α
          prev_connect : r (p.toFun i.castSucc) a
          connect_next : r a (p.toFun i.succ)
          m : Fin (HAdd.hAdd p.length 1)
          x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
          y : α := i.succ.castSucc.insertNth a p.toFun m.succ
          hm : Eq (HAdd.hAdd (↑i) 1) ↑m
          ⊢ r x y
        -/
      · convert connect_next
          /-
            case h.e'_1
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq x a
          -/
        · change Fin.insertNth _ _ _ _ = _
          /-
            case h.e'_1
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (i.succ.castSucc.insertNth a p.toFun m.castSucc) a
          -/
          rw [show m.castSucc = i.succ.castSucc from Fin.ext hm.symm, Fin.insertNth_apply_same]
          /-
            🎉 no goals
          -/
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq y (p.toFun i.succ)
          -/
        · change Fin.insertNth _ _ _ _ = _
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (i.succ.castSucc.insertNth a p.toFun m.succ) (p.toFun i.succ)
          -/
          rw [Fin.insertNth_apply_above]
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.succ.pred ⋯))) (p.toFun i.succ)
          -/
          swap
            /-
              case h.e'_2.h
              α : Type u_1
              r : Rel α α
              β : Type u_2
              s : Rel β β
              p : RelSeries r
              i : Fin p.length
              a : α
              prev_connect : r (p.toFun i.castSucc) a
              connect_next : r a (p.toFun i.succ)
              m : Fin (HAdd.hAdd p.length 1)
              x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
              y : α := i.succ.castSucc.insertNth a p.toFun m.succ
              hm : Eq (HAdd.hAdd (↑i) 1) ↑m
              ⊢ LT.lt i.succ.castSucc m.succ
            -/
          · change i.1 + 1 < m.1 + 1; omega
                                      /-
                                        🎉 no goals
                                      -/
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (Eq.recOn ⋯ (p.toFun (m.succ.pred ⋯))) (p.toFun i.succ)
          -/
          simp only [Fin.pred_succ, eq_rec_constant]
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p : RelSeries r
            i : Fin p.length
            a : α
            prev_connect : r (p.toFun i.castSucc) a
            connect_next : r a (p.toFun i.succ)
            m : Fin (HAdd.hAdd p.length 1)
            x : α := i.succ.castSucc.insertNth a p.toFun m.castSucc
            y : α := i.succ.castSucc.insertNth a p.toFun m.succ
            hm : Eq (HAdd.hAdd (↑i) 1) ↑m
            ⊢ Eq (p.toFun m) (p.toFun i.succ)
          -/
          congr; ext; exact hm.symm
                      /-
                        🎉 no goals
                      -/


/--
A relation series `a₀ -r→ a₁ -r→ ... -r→ aₙ` of `r` gives a relation series of the reverse of `r`
by reversing the series `aₙ ←r- aₙ₋₁ ←r- ... ←r- a₁ ←r- a₀`.
-/
@[simps length]
def reverse (p : RelSeries r) : RelSeries (fun (a b : α) ↦ r b a) where
  length := p.length
  toFun := p ∘ Fin.rev
  step i := by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin p.length
      ⊢ r (Function.comp p.toFun Fin.rev i.succ) (Function.comp p.toFun Fin.rev i.ca …
    -/
    rw [Function.comp_apply, Function.comp_apply]
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin p.length
      ⊢ r (p.toFun i.succ.rev) (p.toFun i.castSucc.rev)
    -/
    have hi : i.1 + 1 ≤ p.length := by omega
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin p.length
      hi : LE.le (HAdd.hAdd (↑i) 1) p.length
      ⊢ r (p.toFun i.succ.rev) (p.toFun i.castSucc.rev)
    -/
    convert p.step ⟨p.length - (i.1 + 1), Nat.sub_lt_self (by omega) hi⟩
      /-
        case h.e'_1.h.e'_4
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        hi : LE.le (HAdd.hAdd (↑i) 1) p.length
        ⊢ Eq i.succ.rev ⟨HSub.hSub p.length (HAdd.hAdd (↑i) 1), ⋯⟩.castSucc
      -/
    · ext; simp
           /-
             🎉 no goals
           -/
      /-
        case h.e'_2.h.e'_4
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        hi : LE.le (HAdd.hAdd (↑i) 1) p.length
        ⊢ Eq i.castSucc.rev ⟨HSub.hSub p.length (HAdd.hAdd (↑i) 1), ⋯⟩.succ
      -/
    · ext
      /-
        case h.e'_2.h.e'_4.h
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        hi : LE.le (HAdd.hAdd (↑i) 1) p.length
        ⊢ Eq ↑i.castSucc.rev ↑⟨HSub.hSub p.length (HAdd.hAdd (↑i) 1), ⋯⟩.succ
      -/
      simp only [Fin.val_rev, Fin.coe_castSucc, Fin.val_succ]
      /-
        case h.e'_2.h.e'_4.h
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p : RelSeries r
        i : Fin p.length
        hi : LE.le (HAdd.hAdd (↑i) 1) p.length
        ⊢ Eq (HSub.hSub (HAdd.hAdd p.length 1) (HAdd.hAdd (↑i) 1)) (HAdd.hAdd (HSub.hS …
      -/
      omega
      /-
        🎉 no goals
      -/


@[simp] lemma reverse_apply (p : RelSeries r) (i : Fin (p.length + 1)) :
    p.reverse i = p i.rev := rfl


@[simp] lemma last_reverse (p : RelSeries r) : p.reverse.last = p.head := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    ⊢ Eq p.reverse.last p.head
  -/
  simp [RelSeries.last, RelSeries.head]
  /-
    🎉 no goals
  -/


@[simp] lemma head_reverse (p : RelSeries r) : p.reverse.head = p.last := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    ⊢ Eq p.reverse.head p.last
  -/
  simp [RelSeries.last, RelSeries.head]
  /-
    🎉 no goals
  -/


@[simp] lemma reverse_reverse {r : Rel α α} (p : RelSeries r) : p.reverse.reverse = p := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    ⊢ Eq p.reverse.reverse p
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/--
Given a series `a₀ -r→ a₁ -r→ ... -r→ aₙ` and an `a` such that `a₀ -r→ a` holds, there is
a series of length `n+1`: `a -r→ a₀ -r→ a₁ -r→ ... -r→ aₙ`.
-/
@[simps! length]
def cons (p : RelSeries r) (newHead : α) (rel : r newHead p.head) : RelSeries r :=
  (singleton r newHead).append p rel


@[simp] lemma head_cons (p : RelSeries r) (newHead : α) (rel : r newHead p.head) :
    (p.cons newHead rel).head = newHead := rfl


@[simp] lemma last_cons (p : RelSeries r) (newHead : α) (rel : r newHead p.head) :
    (p.cons newHead rel).last = p.last := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    newHead : α
    rel : r newHead p.head
    ⊢ Eq (p.cons newHead rel).last p.last
  -/
  delta cons
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    newHead : α
    rel : r newHead p.head
    ⊢ Eq ((RelSeries.singleton r newHead).append p rel).last p.last
  -/
  rw [last_append]
  /-
    🎉 no goals
  -/


/--
Given a series `a₀ -r→ a₁ -r→ ... -r→ aₙ` and an `a` such that `aₙ -r→ a` holds, there is
a series of length `n+1`: `a₀ -r→ a₁ -r→ ... -r→ aₙ -r→ a`.
-/
@[simps! length]
def snoc (p : RelSeries r) (newLast : α) (rel : r p.last newLast) : RelSeries r :=
  p.append (singleton r newLast) rel


@[simp] lemma head_snoc (p : RelSeries r) (newLast : α) (rel : r p.last newLast) :
    (p.snoc newLast rel).head = p.head := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    newLast : α
    rel : r p.last newLast
    ⊢ Eq (p.snoc newLast rel).head p.head
  -/
  delta snoc; rw [head_append]
              /-
                🎉 no goals
              -/


@[simp] lemma last_snoc (p : RelSeries r) (newLast : α) (rel : r p.last newLast) :
    (p.snoc newLast rel).last = newLast := last_append _ _ _

-- This lemma is useful because `last_snoc` is about `Fin.last (p.snoc _ _).length`, but we often
-- see `Fin.last (p.length + 1)` in practice. They are equal by definition, but sometimes simplifier
-- does not pick up `last_snoc`

@[simp] lemma last_snoc' (p : RelSeries r) (newLast : α) (rel : r p.last newLast) :
    p.snoc newLast rel (Fin.last (p.length + 1)) = newLast := last_append _ _ _


@[simp] lemma snoc_castSucc (s : RelSeries r) (a : α) (connect : r s.last a)
    (i : Fin (s.length + 1)) : snoc s a connect (Fin.castSucc i) = s i :=
  Fin.append_left _ _ i


lemma mem_snoc {p : RelSeries r} {newLast : α} {rel : r p.last newLast} {x : α} :
    x ∈ p.snoc newLast rel ↔ x ∈ p ∨ x = newLast := by
  simp only [snoc, append, singleton_length, Nat.add_zero, Nat.reduceAdd, Fin.cast_refl,
    Function.comp_id, mem_def, id_eq, Set.mem_range]
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    newLast : α
    rel : r p.last newLast
    x : α
    ⊢ Iff (Exists fun y => Eq (Fin.append p.toFun (RelSeries.singleton r newLast). …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      r : Rel α α
      p : RelSeries r
      newLast : α
      rel : r p.last newLast
      x : α
      ⊢ (Exists fun y => Eq (Fin.append p.toFun (RelSeries.singleton r newLast).toFu …
    -/
  · rintro ⟨i, rfl⟩
    exact Fin.lastCases (Or.inr <| Fin.append_right _ _ 0) (fun i => Or.inl ⟨⟨i.1, i.2⟩,
      (Fin.append_left _ _ _).symm⟩) i
    /-
      case mpr
      α : Type u_1
      r : Rel α α
      p : RelSeries r
      newLast : α
      rel : r p.last newLast
      x : α
      ⊢ Or (Exists fun y => Eq (p.toFun y) x) (Eq x newLast) → Exists fun y => Eq (F …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      r : Rel α α
      p : RelSeries r
      newLast : α
      rel : r p.last newLast
      x : α
      h : Or (Exists fun y => Eq (p.toFun y) x) (Eq x newLast)
      ⊢ Exists fun y => Eq (Fin.append p.toFun (RelSeries.singleton r newLast).toFun …
    -/
    rcases h with (⟨i, rfl⟩ | rfl)
      /-
        case mpr.inl.intro
        α : Type u_1
        r : Rel α α
        p : RelSeries r
        newLast : α
        rel : r p.last newLast
        i : Fin (HAdd.hAdd p.length 1)
        ⊢ Exists fun y => Eq (Fin.append p.toFun (RelSeries.singleton r newLast).toFun …
      -/
    · exact ⟨i.castSucc, Fin.append_left _ _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u_1
        r : Rel α α
        p : RelSeries r
        x : α
        rel : r p.last x
        ⊢ Exists fun y => Eq (Fin.append p.toFun (RelSeries.singleton r x).toFun y) x
      -/
    · exact ⟨Fin.last _, Fin.append_right _ _ 0⟩
      /-
        🎉 no goals
      -/


/--
If a series `a₀ -r→ a₁ -r→ ...` has positive length, then `a₁ -r→ ...` is another series
-/
@[simps]
def tail (p : RelSeries r) (len_pos : p.length ≠ 0) : RelSeries r where
  length := p.length - 1
  toFun := Fin.tail p ∘ (Fin.cast <| Nat.succ_pred_eq_of_pos <| Nat.pos_of_ne_zero len_pos)
  step i := p.step ⟨i.1 + 1, Nat.lt_pred_iff.mp i.2⟩


@[simp] lemma head_tail (p : RelSeries r) (len_pos : p.length ≠ 0) :
    (p.tail len_pos).head = p 1 := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (p.tail len_pos).head (p.toFun 1)
  -/
  show p (Fin.succ _) = p 1
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (p.toFun (Fin.cast ⋯ 0).succ) (p.toFun 1)
  -/
  congr
  /-
    case e_a
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (Fin.cast ⋯ 0).succ 1
  -/
  ext
  /-
    case e_a.h
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq ↑(Fin.cast ⋯ 0).succ ↑1
  -/
  show (1 : ℕ) = (1 : ℕ) % _
  /-
    case e_a.h
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq 1 (HMod.hMod 1 (HAdd.hAdd p.length 1))
  -/
  rw [Nat.mod_eq_of_lt]
  /-
    case e_a.h
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ LT.lt 1 (HAdd.hAdd p.length 1)
  -/
  simpa only [lt_add_iff_pos_left, Nat.pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


@[simp] lemma last_tail (p : RelSeries r) (len_pos : p.length ≠ 0) :
    (p.tail len_pos).last = p.last := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (p.tail len_pos).last p.last
  -/
  show p _ = p _
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (p.toFun (Fin.cast ⋯ (Fin.last (p.tail len_pos).length)).succ) (p.toFun ( …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (Fin.cast ⋯ (Fin.last (p.tail len_pos).length)).succ (Fin.last p.length)
  -/
  ext
  /-
    case e_a.h
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq ↑(Fin.cast ⋯ (Fin.last (p.tail len_pos).length)).succ ↑(Fin.last p.length)
  -/
  simp only [tail_length, Fin.val_succ, Fin.coe_cast, Fin.val_last]
  /-
    case e_a.h
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    len_pos : Ne p.length 0
    ⊢ Eq (HAdd.hAdd (HSub.hSub p.length 1) 1) p.length
  -/
  exact Nat.succ_pred_eq_of_pos (by simpa [Nat.pos_iff_ne_zero] using len_pos)
  /-
    🎉 no goals
  -/



/--
If a series ``a₀ -r→ a₁ -r→ ... -r→ aₙ``, then `a₀ -r→ a₁ -r→ ... -r→ aₙ₋₁` is
another series -/
@[simps]
def eraseLast (p : RelSeries r) : RelSeries r where
  length := p.length - 1
  toFun i := p ⟨i, lt_of_lt_of_le i.2 (Nat.succ_le_succ tsub_le_self)⟩
  step i := p.step ⟨i, lt_of_lt_of_le i.2 tsub_le_self⟩


@[simp] lemma head_eraseLast (p : RelSeries r) : p.eraseLast.head = p.head := rfl


@[simp] lemma last_eraseLast (p : RelSeries r) :
    p.eraseLast.last = p ⟨p.length.pred, Nat.lt_succ_iff.2 (Nat.pred_le _)⟩ := rfl


/-- In a non-trivial series `p`, the last element of `p.eraseLast` is related to `p.last` -/
lemma eraseLast_last_rel_last (p : RelSeries r) (h : p.length ≠ 0) :
    r p.eraseLast.last p.last := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    h : Ne p.length 0
    ⊢ r p.eraseLast.last p.last
  -/
  simp only [last, Fin.last, eraseLast_length, eraseLast_toFun]
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    h : Ne p.length 0
    ⊢ r (p.toFun ⟨HSub.hSub p.length 1, ⋯⟩) (p.toFun ⟨p.length, ⋯⟩)
  -/
  convert p.step ⟨p.length - 1, by omega⟩
  /-
    case h.e'_2.h.e'_4.h.e'_2
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    h : Ne p.length 0
    ⊢ Eq p.length ↑⟨HSub.hSub p.length 1, ⋯⟩.succ
  -/
  simp only [Nat.succ_eq_add_one, Fin.succ_mk]; omega
                                                /-
                                                  🎉 no goals
                                                -/


/--
Given two series of the form `a₀ -r→ ... -r→ X` and `X -r→ b ---> ...`,
then `a₀ -r→ ... -r→ X -r→ b ...` is another series obtained by combining the given two.
-/
@[simps]
def smash (p q : RelSeries r) (connect : p.last = q.head) : RelSeries r where
  length := p.length + q.length
  toFun i :=
    if H : i.1 < p.length
    then p ⟨i.1, H.trans (lt_add_one _)⟩
    else q ⟨i.1 - p.length,
                                    /-
                                      α : Type u_1
                                      r : Rel α α
                                      β : Type u_2
                                      s : Rel β β
                                      p q : RelSeries r
                                      connect : Eq p.last q.head
                                      i : Fin (HAdd.hAdd (HAdd.hAdd p.length q.length) 1)
                                      H : Not (LT.lt (↑i) p.length)
                                      ⊢ LE.le p.length ↑i
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
      Nat.sub_lt_left_of_lt_add (by rwa [not_lt] at H) (by rw [← add_assoc]; exact i.2)⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  step i := by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p q : RelSeries r
      connect : Eq p.last q.head
      i : Fin (HAdd.hAdd p.length q.length)
      ⊢ r ((fun i => dite (LT.lt (↑i) p.length) (fun H => p.toFun ⟨↑i, ⋯⟩) fun H =>  …
    -/
    dsimp only
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p q : RelSeries r
      connect : Eq p.last q.head
      i : Fin (HAdd.hAdd p.length q.length)
      ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
    -/
    by_cases h₂ : i.1 + 1 < p.length
      /-
        case pos
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : Eq p.last q.head
        i : Fin (HAdd.hAdd p.length q.length)
        h₂ : LT.lt (HAdd.hAdd (↑i) 1) p.length
        ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
      -/
    · have h₁ : i.1 < p.length := lt_trans (lt_add_one _) h₂
      /-
        case pos
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : Eq p.last q.head
        i : Fin (HAdd.hAdd p.length q.length)
        h₂ : LT.lt (HAdd.hAdd (↑i) 1) p.length
        h₁ : LT.lt (↑i) p.length
        ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
      -/
      erw [dif_pos h₁, dif_pos h₂]
      /-
        case pos
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : Eq p.last q.head
        i : Fin (HAdd.hAdd p.length q.length)
        h₂ : LT.lt (HAdd.hAdd (↑i) 1) p.length
        h₁ : LT.lt (↑i) p.length
        ⊢ r (p.toFun ⟨↑i.castSucc, ⋯⟩) (p.toFun ⟨↑i.succ, ⋯⟩)
      -/
      convert p.step ⟨i, h₁⟩ using 1
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : Eq p.last q.head
        i : Fin (HAdd.hAdd p.length q.length)
        h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
      -/
    · erw [dif_neg h₂]
      /-
        case neg
        α : Type u_1
        r : Rel α α
        β : Type u_2
        s : Rel β β
        p q : RelSeries r
        connect : Eq p.last q.head
        i : Fin (HAdd.hAdd p.length q.length)
        h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
      -/
      by_cases h₁ : i.1 < p.length
        /-
          case pos
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : Eq p.last q.head
          i : Fin (HAdd.hAdd p.length q.length)
          h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
          h₁ : LT.lt (↑i) p.length
          ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
        -/
      · erw [dif_pos h₁]
        /-
          case pos
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : Eq p.last q.head
          i : Fin (HAdd.hAdd p.length q.length)
          h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
          h₁ : LT.lt (↑i) p.length
          ⊢ r (p.toFun ⟨↑i.castSucc, ⋯⟩) (q.toFun ⟨HSub.hSub (↑i.succ) p.length, ⋯⟩)
        -/
        have h₃ : p.length = i.1 + 1 := by omega
        /-
          case pos
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : Eq p.last q.head
          i : Fin (HAdd.hAdd p.length q.length)
          h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
          h₁ : LT.lt (↑i) p.length
          h₃ : Eq p.length (HAdd.hAdd (↑i) 1)
          ⊢ r (p.toFun ⟨↑i.castSucc, ⋯⟩) (q.toFun ⟨HSub.hSub (↑i.succ) p.length, ⋯⟩)
        -/
        convert p.step ⟨i, h₁⟩ using 1
        /-
          case h.e'_2
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : Eq p.last q.head
          i : Fin (HAdd.hAdd p.length q.length)
          h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
          h₁ : LT.lt (↑i) p.length
          h₃ : Eq p.length (HAdd.hAdd (↑i) 1)
          ⊢ Eq (q.toFun ⟨HSub.hSub (↑i.succ) p.length, ⋯⟩) (p.toFun ⟨↑i, h₁⟩.succ)
        -/
        convert connect.symm
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : LT.lt (↑i) p.length
            h₃ : Eq p.length (HAdd.hAdd (↑i) 1)
            ⊢ Eq (q.toFun ⟨HSub.hSub (↑i.succ) p.length, ⋯⟩) q.head
          -/
        · aesop
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : LT.lt (↑i) p.length
            h₃ : Eq p.length (HAdd.hAdd (↑i) 1)
            ⊢ Eq (p.toFun ⟨↑i, h₁⟩.succ) p.last
          -/
        · congr; aesop
                 /-
                   🎉 no goals
                 -/
        /-
          case neg
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : Eq p.last q.head
          i : Fin (HAdd.hAdd p.length q.length)
          h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
          h₁ : Not (LT.lt (↑i) p.length)
          ⊢ r (dite (LT.lt (↑i.castSucc) p.length) (fun H => p.toFun ⟨↑i.castSucc, ⋯⟩) f …
        -/
      · erw [dif_neg h₁]
        /-
          case neg
          α : Type u_1
          r : Rel α α
          β : Type u_2
          s : Rel β β
          p q : RelSeries r
          connect : Eq p.last q.head
          i : Fin (HAdd.hAdd p.length q.length)
          h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
          h₁ : Not (LT.lt (↑i) p.length)
          ⊢ r (q.toFun ⟨HSub.hSub (↑i.castSucc) p.length, ⋯⟩) (q.toFun ⟨HSub.hSub (↑i.su …
        -/
        convert q.step ⟨i.1 - p.length, _⟩ using 1
          /-
            case h.e'_2
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : Not (LT.lt (↑i) p.length)
            ⊢ Eq (q.toFun ⟨HSub.hSub (↑i.succ) p.length, ⋯⟩) (q.toFun ⟨HSub.hSub (↑i) p.le …
          -/
        · congr
          /-
            case h.e'_2.e_a.e_val
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : Not (LT.lt (↑i) p.length)
            ⊢ Eq (HSub.hSub (↑i.succ) p.length) (HAdd.hAdd (HSub.hSub (↑i) p.length) 1)
          -/
          change (i.1 + 1) - _ = _
          /-
            case h.e'_2.e_a.e_val
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : Not (LT.lt (↑i) p.length)
            ⊢ Eq (HSub.hSub (HAdd.hAdd (↑i) 1) p.length) (HAdd.hAdd (HSub.hSub (↑i) p.leng …
          -/
          rw [Nat.sub_add_comm]
          /-
            case h.e'_2.e_a.e_val
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : Not (LT.lt (↑i) p.length)
            ⊢ LE.le p.length ↑i
          -/
          rwa [not_lt] at h₁
          /-
            🎉 no goals
          -/
          /-
            case neg
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : Not (LT.lt (↑i) p.length)
            ⊢ LT.lt (HSub.hSub (↑i) p.length) q.length
          -/
        · refine Nat.sub_lt_left_of_lt_add ?_ i.2
          /-
            case neg
            α : Type u_1
            r : Rel α α
            β : Type u_2
            s : Rel β β
            p q : RelSeries r
            connect : Eq p.last q.head
            i : Fin (HAdd.hAdd p.length q.length)
            h₂ : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
            h₁ : Not (LT.lt (↑i) p.length)
            ⊢ LE.le p.length ↑i
          -/
          rwa [not_lt] at h₁
          /-
            🎉 no goals
          -/


lemma smash_castAdd {p q : RelSeries r} (connect : p.last = q.head) (i : Fin p.length) :
    p.smash q connect (Fin.castSucc <| i.castAdd q.length) = p (Fin.castSucc i) := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : Eq p.last q.head
    i : Fin p.length
    ⊢ Eq ((p.smash q connect).toFun (Fin.castAdd q.length i).castSucc) (p.toFun i. …
  -/
  unfold smash
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : Eq p.last q.head
    i : Fin p.length
    ⊢ Eq ({ length := HAdd.hAdd p.length q.length, toFun := fun i => dite (LT.lt ( …
  -/
  dsimp
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : Eq p.last q.head
    i : Fin p.length
    ⊢ Eq (dite (LT.lt (↑i) p.length) (fun H => p.toFun ⟨↑i, ⋯⟩) fun H => q.toFun ⟨ …
  -/
  rw [dif_pos i.2]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    connect : Eq p.last q.head
    i : Fin p.length
    ⊢ Eq (p.toFun ⟨↑i, ⋯⟩) (p.toFun i.castSucc)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma smash_succ_castAdd {p q : RelSeries r} (h : p.last = q.head)
    (i : Fin p.length) : p.smash q h (i.castAdd q.length).succ = p i.succ := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin p.length
    ⊢ Eq ((p.smash q h).toFun (Fin.castAdd q.length i).succ) (p.toFun i.succ)
  -/
  rw [smash_toFun]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin p.length
    ⊢ Eq (dite (LT.lt (↑(Fin.castAdd q.length i).succ) p.length) (fun H => p.toFun …
  -/
  split_ifs with H
    /-
      case pos
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin p.length
      H : LT.lt (↑(Fin.castAdd q.length i).succ) p.length
      ⊢ Eq (p.toFun ⟨↑(Fin.castAdd q.length i).succ, ⋯⟩) (p.toFun i.succ)
    -/
  · congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin p.length
      H : Not (LT.lt (↑(Fin.castAdd q.length i).succ) p.length)
      ⊢ Eq (q.toFun ⟨HSub.hSub (↑(Fin.castAdd q.length i).succ) p.length, ⋯⟩) (p.toF …
    -/
  · simp only [Fin.val_succ, Fin.coe_castAdd] at H
    /-
      case neg
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin p.length
      H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
      ⊢ Eq (q.toFun ⟨HSub.hSub (↑(Fin.castAdd q.length i).succ) p.length, ⋯⟩) (p.toF …
    -/
    convert h.symm
      /-
        case h.e'_2
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ Eq (q.toFun ⟨HSub.hSub (↑(Fin.castAdd q.length i).succ) p.length, ⋯⟩) q.head
      -/
    · congr
      /-
        case h.e'_2.e_a.e_val
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ Eq (HSub.hSub (↑(Fin.castAdd q.length i).succ) p.length) (HMod.hMod 0 (HAdd. …
      -/
      simp only [Fin.val_succ, Fin.coe_castAdd, Nat.zero_mod, tsub_eq_zero_iff_le]
      /-
        case h.e'_2.e_a.e_val
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ LE.le (HAdd.hAdd (↑i) 1) p.length
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ Eq (p.toFun i.succ) p.last
      -/
    · congr
      /-
        case h.e'_3.e_a
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ Eq i.succ (Fin.last p.length)
      -/
      ext
      /-
        case h.e'_3.e_a.h
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ Eq ↑i.succ ↑(Fin.last p.length)
      -/
      change i.1 + 1 = p.length
      /-
        case h.e'_3.e_a.h
        α : Type u_1
        r : Rel α α
        p q : RelSeries r
        h : Eq p.last q.head
        i : Fin p.length
        H : Not (LT.lt (HAdd.hAdd (↑i) 1) p.length)
        ⊢ Eq (HAdd.hAdd (↑i) 1) p.length
      -/
      omega
      /-
        🎉 no goals
      -/


lemma smash_natAdd {p q : RelSeries r} (h : p.last = q.head) (i : Fin q.length) :
    smash p q h (Fin.castSucc <| i.natAdd p.length) = q (Fin.castSucc i) := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin q.length
    ⊢ Eq ((p.smash q h).toFun (Fin.natAdd p.length i).castSucc) (q.toFun i.castSucc)
  -/
  rw [smash_toFun, dif_neg (by simp)]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin q.length
    ⊢ Eq (q.toFun ⟨HSub.hSub (↑(Fin.natAdd p.length i).castSucc) p.length, ⋯⟩) (q. …
  -/
  congr
  /-
    case e_a.e_val
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin q.length
    ⊢ Eq (HSub.hSub (↑(Fin.natAdd p.length i).castSucc) p.length) ↑i
  -/
  exact Nat.add_sub_self_left _ _
  /-
    🎉 no goals
  -/


lemma smash_succ_natAdd {p q : RelSeries r} (h : p.last = q.head) (i : Fin q.length) :
    smash p q h (i.natAdd p.length).succ = q i.succ := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin q.length
    ⊢ Eq ((p.smash q h).toFun (Fin.natAdd p.length i).succ) (q.toFun i.succ)
  -/
  rw [smash_toFun]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    i : Fin q.length
    ⊢ Eq (dite (LT.lt (↑(Fin.natAdd p.length i).succ) p.length) (fun H => p.toFun  …
  -/
  split_ifs with H
    /-
      case pos
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin q.length
      H : LT.lt (↑(Fin.natAdd p.length i).succ) p.length
      ⊢ Eq (p.toFun ⟨↑(Fin.natAdd p.length i).succ, ⋯⟩) (q.toFun i.succ)
    -/
  · have H' : p.length < p.length + (i.1 + 1) := by omega
    /-
      case pos
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin q.length
      H : LT.lt (↑(Fin.natAdd p.length i).succ) p.length
      H' : LT.lt p.length (HAdd.hAdd p.length (HAdd.hAdd (↑i) 1))
      ⊢ Eq (p.toFun ⟨↑(Fin.natAdd p.length i).succ, ⋯⟩) (q.toFun i.succ)
    -/
    exact (lt_irrefl _ (H.trans H')).elim
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin q.length
      H : Not (LT.lt (↑(Fin.natAdd p.length i).succ) p.length)
      ⊢ Eq (q.toFun ⟨HSub.hSub (↑(Fin.natAdd p.length i).succ) p.length, ⋯⟩) (q.toFu …
    -/
  · congr
    /-
      case neg.e_a.e_val
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin q.length
      H : Not (LT.lt (↑(Fin.natAdd p.length i).succ) p.length)
      ⊢ Eq (HSub.hSub (↑(Fin.natAdd p.length i).succ) p.length) (HAdd.hAdd (↑i) 1)
    -/
    simp only [Fin.val_succ, Fin.coe_natAdd]
    /-
      case neg.e_a.e_val
      α : Type u_1
      r : Rel α α
      p q : RelSeries r
      h : Eq p.last q.head
      i : Fin q.length
      H : Not (LT.lt (↑(Fin.natAdd p.length i).succ) p.length)
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd p.length ↑i) 1) p.length) (HAdd.hAdd (↑i …
    -/
    rw [add_assoc, Nat.add_sub_cancel_left]
    /-
      🎉 no goals
    -/


@[simp] lemma head_smash {p q : RelSeries r} (h : p.last = q.head) :
    (smash p q h).head = p.head := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    ⊢ Eq (p.smash q h).head p.head
  -/
  delta head smash
  simp only [Fin.val_zero, Fin.zero_eta, zero_le, tsub_eq_zero_of_le, dite_eq_ite,
    ite_eq_left_iff, not_lt, nonpos_iff_eq_zero]
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    ⊢ Eq p.length 0 → Eq (q.toFun 0) (p.toFun 0)
  -/
  intro H; convert h.symm; congr; aesop
                                  /-
                                    🎉 no goals
                                  -/


@[simp] lemma last_smash {p q : RelSeries r} (h : p.last = q.head) :
    (smash p q h).last = q.last := by
  /-
    α : Type u_1
    r : Rel α α
    p q : RelSeries r
    h : Eq p.last q.head
    ⊢ Eq (p.smash q h).last q.last
  -/
  delta smash last; aesop
                    /-
                      🎉 no goals
                    -/


/-- Given the series `a₀ -r→ … -r→ aᵢ -r→ … -r→ aₙ`, the series `a₀ -r→ … -r→ aᵢ`. -/
@[simps! length]
def take {r : Rel α α} (p : RelSeries r) (i : Fin (p.length + 1)) : RelSeries r where
  length := i
                                        /-
                                          α : Type u_1
                                          r✝ : Rel α α
                                          β : Type u_2
                                          s : Rel β β
                                          r : Rel α α
                                          p : RelSeries r
                                          i : Fin (HAdd.hAdd p.length 1)
                                          x✝ : Fin (HAdd.hAdd (↑i) 1)
                                          j : Nat
                                          h : LT.lt j (HAdd.hAdd (↑i) 1)
                                          ⊢ LT.lt j (HAdd.hAdd p.length 1)
                                        -/
  toFun := fun ⟨j, h⟩ => p.toFun ⟨j, by omega⟩
                                        /-
                                          🎉 no goals
                                        -/
                                      /-
                                        α : Type u_1
                                        r✝ : Rel α α
                                        β : Type u_2
                                        s : Rel β β
                                        r : Rel α α
                                        p : RelSeries r
                                        i : Fin (HAdd.hAdd p.length 1)
                                        x✝ : Fin ↑i
                                        j : Nat
                                        h : LT.lt j ↑i
                                        ⊢ LT.lt j p.length
                                      -/
  step := fun ⟨j, h⟩ => p.step ⟨j, by omega⟩
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
lemma head_take (p : RelSeries r) (i : Fin (p.length + 1)) :
                                   /-
                                     α : Type u_1
                                     r : Rel α α
                                     p : RelSeries r
                                     i : Fin (HAdd.hAdd p.length 1)
                                     ⊢ Eq (p.take i).head p.head
                                   -/
    (p.take i).head = p.head := by simp [take, head]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma last_take (p : RelSeries r) (i : Fin (p.length + 1)) :
                                /-
                                  α : Type u_1
                                  r : Rel α α
                                  p : RelSeries r
                                  i : Fin (HAdd.hAdd p.length 1)
                                  ⊢ Eq (p.take i).last (p.toFun i)
                                -/
    (p.take i).last = p i := by simp [take, last, Fin.last]
                                /-
                                  🎉 no goals
                                -/


/-- Given the series `a₀ -r→ … -r→ aᵢ -r→ … -r→ aₙ`, the series `aᵢ₊₁ -r→ … -r→ aᵢ`. -/
@[simps! length]
def drop (p : RelSeries r) (i : Fin (p.length + 1)) : RelSeries r where
  length := p.length - i
                                          /-
                                            α : Type u_1
                                            r : Rel α α
                                            β : Type u_2
                                            s : Rel β β
                                            p : RelSeries r
                                            i : Fin (HAdd.hAdd p.length 1)
                                            x✝ : Fin (HAdd.hAdd (HSub.hSub p.length ↑i) 1)
                                            j : Nat
                                            h : LT.lt j (HAdd.hAdd (HSub.hSub p.length ↑i) 1)
                                            ⊢ LT.lt (HAdd.hAdd j ↑i) (HAdd.hAdd p.length 1)
                                          -/
  toFun := fun ⟨j, h⟩ => p.toFun ⟨j+i, by omega⟩
                                          /-
                                            🎉 no goals
                                          -/
  step := fun ⟨j, h⟩ => by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin (HAdd.hAdd p.length 1)
      x✝ : Fin (HSub.hSub p.length ↑i)
      j : Nat
      h : LT.lt j (HSub.hSub p.length ↑i)
      ⊢ r ((fun x => RelSeries.drop.match_1 p i (fun x => α) x fun j h => p.toFun ⟨H …
    -/
    convert p.step ⟨j+i.1, by omega⟩
    /-
      case h.e'_2.h.e'_4.h.h.h.e'_2
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      p : RelSeries r
      i : Fin (HAdd.hAdd p.length 1)
      x✝ : Fin (HSub.hSub p.length ↑i)
      j : Nat
      h : LT.lt j (HSub.hSub p.length ↑i)
      ⊢ Eq (HAdd.hAdd ↑⟨j, h⟩.succ ↑i) ↑⟨HAdd.hAdd j ↑i, ⋯⟩.succ
    -/
    simp only [Nat.succ_eq_add_one, Fin.succ_mk]; omega
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
lemma head_drop (p : RelSeries r) (i : Fin (p.length + 1)) : (p.drop i).head = p.toFun i := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (p.drop i).head (p.toFun i)
  -/
  simp [drop, head]
  /-
    🎉 no goals
  -/


@[simp]
lemma last_drop (p : RelSeries r) (i : Fin (p.length + 1)) : (p.drop i).last = p.last := by
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (p.drop i).last p.last
  -/
  simp only [last, drop, Fin.last]
  /-
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (p.toFun ⟨HAdd.hAdd (HSub.hSub p.length ↑i) ↑i, ⋯⟩) (p.toFun ⟨p.length, ⋯⟩)
  -/
  congr
  /-
    case e_a.e_val
    α : Type u_1
    r : Rel α α
    p : RelSeries r
    i : Fin (HAdd.hAdd p.length 1)
    ⊢ Eq (HAdd.hAdd (HSub.hSub p.length ↑i) ↑i) p.length
  -/
  omega
  /-
    🎉 no goals
  -/


variable {r} in
lemma Rel.not_finiteDimensional_iff [Nonempty α] :
    ¬ r.FiniteDimensional ↔ r.InfiniteDimensional := by
  /-
    α : Type u_1
    r : Rel α α
    inst✝ : Nonempty α
    ⊢ Iff (Not r.FiniteDimensional) r.InfiniteDimensional
  -/
  rw [finiteDimensional_iff, infiniteDimensional_iff]
  /-
    α : Type u_1
    r : Rel α α
    inst✝ : Nonempty α
    ⊢ Iff (Not (Exists fun x => ∀ (y : RelSeries r), LE.le y.length x.length)) (∀  …
  -/
  push_neg
  /-
    α : Type u_1
    r : Rel α α
    inst✝ : Nonempty α
    ⊢ Iff (∀ (x : RelSeries r), Exists fun y => LT.lt x.length y.length) (∀ (n : N …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      r : Rel α α
      inst✝ : Nonempty α
      ⊢ (∀ (x : RelSeries r), Exists fun y => LT.lt x.length y.length) → ∀ (n : Nat) …
    -/
  · intro H n
    induction n with
    | zero => refine ⟨⟨0, ![Nonempty.some ‹_›], by simp⟩, by simp⟩
    | succ n IH =>
      obtain ⟨l, hl⟩ := IH
      obtain ⟨l', hl'⟩ := H l
      exact ⟨l'.take ⟨n + 1, by simpa [hl] using hl'⟩, rfl⟩
    /-
      case mpr
      α : Type u_1
      r : Rel α α
      inst✝ : Nonempty α
      ⊢ (∀ (n : Nat), Exists fun x => Eq x.length n) → ∀ (x : RelSeries r), Exists f …
    -/
  · intro H l
    /-
      case mpr
      α : Type u_1
      r : Rel α α
      inst✝ : Nonempty α
      H : ∀ (n : Nat), Exists fun x => Eq x.length n
      l : RelSeries r
      ⊢ Exists fun y => LT.lt l.length y.length
    -/
    obtain ⟨l', hl'⟩ := H (l.length + 1)
    /-
      case mpr.intro
      α : Type u_1
      r : Rel α α
      inst✝ : Nonempty α
      H : ∀ (n : Nat), Exists fun x => Eq x.length n
      l l' : RelSeries r
      hl' : Eq l'.length (HAdd.hAdd l.length 1)
      ⊢ Exists fun y => LT.lt l.length y.length
    -/
    exact ⟨l', by simp [hl']⟩
    /-
      🎉 no goals
    -/


variable {r} in
lemma Rel.not_infiniteDimensional_iff [Nonempty α] :
    ¬ r.InfiniteDimensional ↔ r.FiniteDimensional := by
  /-
    α : Type u_1
    r : Rel α α
    inst✝ : Nonempty α
    ⊢ Iff (Not r.InfiniteDimensional) r.FiniteDimensional
  -/
  rw [← not_finiteDimensional_iff, not_not]
  /-
    🎉 no goals
  -/


lemma Rel.finiteDimensional_or_infiniteDimensional [Nonempty α] :
    r.FiniteDimensional ∨ r.InfiniteDimensional := by
  /-
    α : Type u_1
    r : Rel α α
    inst✝ : Nonempty α
    ⊢ Or r.FiniteDimensional r.InfiniteDimensional
  -/
  rw [← not_finiteDimensional_iff]
  /-
    α : Type u_1
    r : Rel α α
    inst✝ : Nonempty α
    ⊢ Or r.FiniteDimensional (Not r.FiniteDimensional)
  -/
  exact em r.FiniteDimensional
  /-
    🎉 no goals
  -/


/-- A type is finite dimensional if its `LTSeries` has bounded length. -/
abbrev FiniteDimensionalOrder (γ : Type*) [Preorder γ] :=
  Rel.FiniteDimensional ((· < ·) : γ → γ → Prop)


instance FiniteDimensionalOrder.ofUnique (γ : Type*) [Preorder γ] [Unique γ] :
    FiniteDimensionalOrder γ where
  exists_longest_relSeries := ⟨.singleton _ default, fun x ↦ by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      γ : Type u_3
      inst✝¹ : Preorder γ
      inst✝ : Unique γ
      x : RelSeries fun x1 x2 => LT.lt x1 x2
      ⊢ LE.le x.length (RelSeries.singleton (fun x1 x2 => LT.lt x1 x2) Inhabited.def …
    -/
    by_contra! r
    /-
      α : Type u_1
      r✝ : Rel α α
      β : Type u_2
      s : Rel β β
      γ : Type u_3
      inst✝¹ : Preorder γ
      inst✝ : Unique γ
      x : RelSeries fun x1 x2 => LT.lt x1 x2
      r : LT.lt (RelSeries.singleton (fun x1 x2 => LT.lt x1 x2) Inhabited.default).l …
      ⊢ False
    -/
    exact (ne_of_lt <| x.step ⟨0, by omega⟩) <| Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/


/-- A type is infinite dimensional if it has `LTSeries` of at least arbitrary length -/
abbrev InfiniteDimensionalOrder (γ : Type*) [Preorder γ] :=
  Rel.InfiniteDimensional ((· < ·) : γ → γ → Prop)


/--
If `α` is a preorder, a LTSeries is a relation series of the less than relation.
-/
abbrev LTSeries := RelSeries ((· < ·) : Rel α α)


/-- The longest `<`-series when a type is finite dimensional -/
protected noncomputable def longestOf [FiniteDimensionalOrder α] : LTSeries α :=
  RelSeries.longestOf _


/-- A `<`-series with length `n` if the relation is infinite dimensional -/
protected noncomputable def withLength [InfiniteDimensionalOrder α] (n : ℕ) : LTSeries α :=
  RelSeries.withLength _ n


@[simp] lemma length_withLength [InfiniteDimensionalOrder α] (n : ℕ) :
    (LTSeries.withLength α n).length = n :=
  RelSeries.length_withLength _ _


/-- if `α` is infinite dimensional, then `α` is nonempty. -/
lemma nonempty_of_infiniteDimensionalType [InfiniteDimensionalOrder α] : Nonempty α :=
  ⟨LTSeries.withLength α 0 0⟩


lemma longestOf_is_longest [FiniteDimensionalOrder α] (x : LTSeries α) :
    x.length ≤ (LTSeries.longestOf α).length :=
  RelSeries.length_le_length_longestOf _ _


lemma longestOf_len_unique [FiniteDimensionalOrder α] (p : LTSeries α)
    (is_longest : ∀ (q : LTSeries α), q.length ≤ p.length) :
    p.length = (LTSeries.longestOf α).length :=
  le_antisymm (longestOf_is_longest _) (is_longest _)



lemma strictMono (x : LTSeries α) : StrictMono x :=
  fun _ _ h => x.rel_of_lt h


lemma monotone (x : LTSeries α) : Monotone x :=
  x.strictMono.monotone


lemma head_le_last (x : LTSeries α) : x.head ≤ x.last :=
  LTSeries.monotone x (Fin.zero_le _)


/-- An alternative constructor of `LTSeries` from a strictly monotone function. -/
@[simps]
def mk (length : ℕ) (toFun : Fin (length + 1) → α) (strictMono : StrictMono toFun) :
    LTSeries α where
  toFun := toFun
  step i := strictMono <| lt_add_one i.1


/-- An injection from the type of strictly monotone functions with limited length to `LTSeries`. -/
def injStrictMono (n : ℕ) :
    {f : (l : Fin n) × (Fin (l + 1) → α) // StrictMono f.2} ↪ LTSeries α where
  toFun f := mk f.1.1 f.1.2 f.2
  inj' f g e := by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      f g : Subtype fun f => StrictMono f.snd
      e : Eq ((fun f => LTSeries.mk (↑(↑f).fst) (↑f).snd ⋯) f) ((fun f => LTSeries.m …
      ⊢ Eq f g
    -/
    obtain ⟨⟨lf, f⟩, mf⟩ := f
    /-
      case mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      g : Subtype fun f => StrictMono f.snd
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono ⟨lf, f⟩.snd
      e : Eq ((fun f => LTSeries.mk (↑(↑f).fst) (↑f).snd ⋯) ⟨⟨lf, f⟩, mf⟩) ((fun f = …
      ⊢ Eq ⟨⟨lf, f⟩, mf⟩ g
    -/
    obtain ⟨⟨lg, g⟩, mg⟩ := g
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono ⟨lf, f⟩.snd
      lg : Fin n
      g : Fin (HAdd.hAdd (↑lg) 1) → α
      mg : StrictMono ⟨lg, g⟩.snd
      e : Eq ((fun f => LTSeries.mk (↑(↑f).fst) (↑f).snd ⋯) ⟨⟨lf, f⟩, mf⟩) ((fun f = …
      ⊢ Eq ⟨⟨lf, f⟩, mf⟩ ⟨⟨lg, g⟩, mg⟩
    -/
    dsimp only at mf mg e
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      lg : Fin n
      g : Fin (HAdd.hAdd (↑lg) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lg) g ⋯)
      ⊢ Eq ⟨⟨lf, f⟩, mf⟩ ⟨⟨lg, g⟩, mg⟩
    -/
    have leq := congr($(e).length)
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      lg : Fin n
      g : Fin (HAdd.hAdd (↑lg) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lg) g ⋯)
      leq : Eq (LTSeries.mk (↑lf) f ⋯).length (LTSeries.mk (↑lg) g ⋯).length
      ⊢ Eq ⟨⟨lf, f⟩, mf⟩ ⟨⟨lg, g⟩, mg⟩
    -/
    rw [mk_length lf f mf, mk_length lg g mg, Fin.val_eq_val] at leq
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      lg : Fin n
      g : Fin (HAdd.hAdd (↑lg) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lg) g ⋯)
      leq : Eq lf lg
      ⊢ Eq ⟨⟨lf, f⟩, mf⟩ ⟨⟨lg, g⟩, mg⟩
    -/
    subst leq
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      g : Fin (HAdd.hAdd (↑lf) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lf) g ⋯)
      ⊢ Eq ⟨⟨lf, f⟩, mf⟩ ⟨⟨lf, g⟩, mg⟩
    -/
    simp_rw [Subtype.mk_eq_mk, Sigma.mk.inj_iff, heq_eq_eq, true_and]
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      g : Fin (HAdd.hAdd (↑lf) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lf) g ⋯)
      ⊢ Eq f g
    -/
    have feq := fun i ↦ congr($(e).toFun i)
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      g : Fin (HAdd.hAdd (↑lf) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lf) g ⋯)
      feq : ∀ (i : Fin (HAdd.hAdd (Mathlib.Tactic.TermCongr.cHole (LTSeries.mk (↑lf) …
      ⊢ Eq f g
    -/
    simp_rw [mk_toFun lf f mf, mk_toFun lf g mg, mk_length lf f mf] at feq
    /-
      case mk.mk.mk.mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      n : Nat
      lf : Fin n
      f : Fin (HAdd.hAdd (↑lf) 1) → α
      mf : StrictMono f
      g : Fin (HAdd.hAdd (↑lf) 1) → α
      mg : StrictMono g
      e : Eq (LTSeries.mk (↑lf) f ⋯) (LTSeries.mk (↑lf) g ⋯)
      feq : ∀ (i : Fin (HAdd.hAdd (↑lf) 1)), Eq (f i) (g i)
      ⊢ Eq f g
    -/
    rwa [funext_iff]
    /-
      🎉 no goals
    -/


/--
For two preorders `α, β`, if `f : α → β` is strictly monotonic, then a strict chain of `α`
can be pushed out to a strict chain of `β` by
`a₀ < a₁ < ... < aₙ ↦ f a₀ < f a₁ < ... < f aₙ`
-/
@[simps!]
def map (p : LTSeries α) (f : α → β) (hf : StrictMono f) : LTSeries β :=
  LTSeries.mk p.length (f.comp p) (hf.comp p.strictMono)


@[simp] lemma head_map (p : LTSeries α) (f : α → β) (hf : StrictMono f) :
  (p.map f hf).head = f p.head := rfl


@[simp] lemma last_map (p : LTSeries α) (f : α → β) (hf : StrictMono f) :
  (p.map f hf).last = f p.last := rfl


/--
For two preorders `α, β`, if `f : α → β` is surjective and strictly comonotonic, then a
strict series of `β` can be pulled back to a strict chain of `α` by
`b₀ < b₁ < ... < bₙ ↦ f⁻¹ b₀ < f⁻¹ b₁ < ... < f⁻¹ bₙ` where `f⁻¹ bᵢ` is an arbitrary element in the
preimage of `f⁻¹ {bᵢ}`.
-/
@[simps!]
noncomputable def comap (p : LTSeries β) (f : α → β)
  (comap : ∀ ⦃x y⦄, f x < f y → x < y)
  (surjective : Function.Surjective f) :
  LTSeries α := mk p.length (fun i ↦ (surjective (p i)).choose)
                           /-
                             α : Type u_1
                             r : Rel α α
                             β : Type u_2
                             s : Rel β β
                             inst✝¹ : Preorder α
                             inst✝ : Preorder β
                             p : LTSeries β
                             f : α → β
                             comap : ∀ ⦃x y : α⦄, LT.lt (f x) (f y) → LT.lt x y
                             surjective : Function.Surjective f
                             i j : Fin (HAdd.hAdd p.length 1)
                             h : LT.lt i j
                             ⊢ LT.lt (f ((fun i => ⋯.choose) i)) (f ((fun i => ⋯.choose) j))
                           -/
    (fun i j h ↦ comap (by simpa only [(surjective _).choose_spec] using p.strictMono h))
                           /-
                             🎉 no goals
                           -/


/-- The strict series `0 < … < n` in `ℕ`. -/
def range (n : ℕ) : LTSeries ℕ where
  length := n
  toFun := fun i => i
  step i := Nat.lt_add_one i


@[simp] lemma length_range (n : ℕ) : (range n).length = n := rfl


@[simp] lemma range_apply (n : ℕ) (i : Fin (n+1)) : (range n) i = i := rfl


@[simp] lemma head_range (n : ℕ) : (range n).head = 0 := rfl


@[simp] lemma last_range (n : ℕ) : (range n).last = n := rfl


/--
In ℕ, two entries in an `LTSeries` differ by at least the difference of their indices.
(Expressed in a way that avoids subtraction).
 -/
lemma apply_add_index_le_apply_add_index_nat (p : LTSeries ℕ) (i j : Fin (p.length + 1))
    (hij : i ≤ j) : p i + j ≤ p j + i := by
  /-
    p : LTSeries Nat
    i j : Fin (HAdd.hAdd p.length 1)
    hij : LE.le i j
    ⊢ LE.le (HAdd.hAdd (p.toFun i) ↑j) (HAdd.hAdd (p.toFun j) ↑i)
  -/
  have ⟨i, hi⟩ := i
  /-
    p : LTSeries Nat
    i✝ j : Fin (HAdd.hAdd p.length 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd p.length 1)
    hij : LE.le ⟨i, hi⟩ j
    ⊢ LE.le (HAdd.hAdd (p.toFun ⟨i, hi⟩) ↑j) (HAdd.hAdd (p.toFun j) ↑⟨i, hi⟩)
  -/
  have ⟨j, hj⟩ := j
  /-
    p : LTSeries Nat
    i✝ j✝ : Fin (HAdd.hAdd p.length 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd p.length 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd p.length 1)
    hij : LE.le ⟨i, hi⟩ ⟨j, hj⟩
    ⊢ LE.le (HAdd.hAdd (p.toFun ⟨i, hi⟩) ↑⟨j, hj⟩) (HAdd.hAdd (p.toFun ⟨j, hj⟩) ↑⟨ …
  -/
  simp only [Fin.mk_le_mk] at hij
  /-
    p : LTSeries Nat
    i✝ j✝ : Fin (HAdd.hAdd p.length 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd p.length 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd p.length 1)
    hij : LE.le i j
    ⊢ LE.le (HAdd.hAdd (p.toFun ⟨i, hi⟩) ↑⟨j, hj⟩) (HAdd.hAdd (p.toFun ⟨j, hj⟩) ↑⟨ …
  -/
  simp only at *
  induction j, hij using Nat.le_induction with
  | base => simp
  | succ j _hij ih =>
    specialize ih (Nat.lt_of_succ_lt hj)
    have step : p ⟨j, _⟩ < p ⟨j + 1, _⟩ := p.step ⟨j, by omega⟩
    norm_cast at *; omega


/--
In ℤ, two entries in an `LTSeries` differ by at least the difference of their indices.
(Expressed in a way that avoids subtraction).
-/
lemma apply_add_index_le_apply_add_index_int (p : LTSeries ℤ) (i j : Fin (p.length + 1))
    (hij : i ≤ j) : p i + j ≤ p j + i := by
  -- The proof is identical to `LTSeries.apply_add_index_le_apply_add_index_nat`, but seemed easier
  -- to copy rather than to abstract
  /-
    p : LTSeries Int
    i j : Fin (HAdd.hAdd p.length 1)
    hij : LE.le i j
    ⊢ LE.le (HAdd.hAdd (p.toFun i) ↑↑j) (HAdd.hAdd (p.toFun j) ↑↑i)
  -/
  have ⟨i, hi⟩ := i
  /-
    p : LTSeries Int
    i✝ j : Fin (HAdd.hAdd p.length 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd p.length 1)
    hij : LE.le ⟨i, hi⟩ j
    ⊢ LE.le (HAdd.hAdd (p.toFun ⟨i, hi⟩) ↑↑j) (HAdd.hAdd (p.toFun j) ↑↑⟨i, hi⟩)
  -/
  have ⟨j, hj⟩ := j
  /-
    p : LTSeries Int
    i✝ j✝ : Fin (HAdd.hAdd p.length 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd p.length 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd p.length 1)
    hij : LE.le ⟨i, hi⟩ ⟨j, hj⟩
    ⊢ LE.le (HAdd.hAdd (p.toFun ⟨i, hi⟩) ↑↑⟨j, hj⟩) (HAdd.hAdd (p.toFun ⟨j, hj⟩) ↑ …
  -/
  simp only [Fin.mk_le_mk] at hij
  /-
    p : LTSeries Int
    i✝ j✝ : Fin (HAdd.hAdd p.length 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd p.length 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd p.length 1)
    hij : LE.le i j
    ⊢ LE.le (HAdd.hAdd (p.toFun ⟨i, hi⟩) ↑↑⟨j, hj⟩) (HAdd.hAdd (p.toFun ⟨j, hj⟩) ↑ …
  -/
  simp only at *
  induction j, hij using Nat.le_induction with
  | base => simp
  | succ j _hij ih =>
    specialize ih (Nat.lt_of_succ_lt hj)
    have step : p ⟨j, _⟩ < p ⟨j + 1, _⟩:= p.step ⟨j, by omega⟩
    norm_cast at *; omega


/-- In ℕ, the head and tail of an `LTSeries` differ at least by the length of the series -/
lemma head_add_length_le_nat (p : LTSeries ℕ) : p.head + p.length ≤ p.last :=
  LTSeries.apply_add_index_le_apply_add_index_nat _ _ (Fin.last _) (Fin.zero_le _)


/-- In ℤ, the head and tail of an `LTSeries` differ at least by the length of the series -/
lemma head_add_length_le_int (p : LTSeries ℤ) : p.head + p.length ≤ p.last := by
  /-
    p : LTSeries Int
    ⊢ LE.le (HAdd.hAdd (RelSeries.head p) ↑p.length) (RelSeries.last p)
  -/
  simpa using LTSeries.apply_add_index_le_apply_add_index_int _ _ (Fin.last _) (Fin.zero_le _)
  /-
    🎉 no goals
  -/


lemma length_lt_card (s : LTSeries α) : s.length < Fintype.card α := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Fintype α
    s : LTSeries α
    ⊢ LT.lt s.length (Fintype.card α)
  -/
  by_contra! h
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Fintype α
    s : LTSeries α
    h : LE.le (Fintype.card α) s.length
    ⊢ False
  -/
  obtain ⟨i, j, hn, he⟩ := Fintype.exists_ne_map_eq_of_card_lt s (by rw [Fintype.card_fin]; omega)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Fintype α
    s : LTSeries α
    h : LE.le (Fintype.card α) s.length
    i j : Fin (HAdd.hAdd s.length 1)
    hn : Ne i j
    he : Eq (s.toFun i) (s.toFun j)
    ⊢ False
  -/
  wlog hl : i < j generalizing i j
    /-
      case intro.intro.intro.inr
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : Fintype α
      s : LTSeries α
      h : LE.le (Fintype.card α) s.length
      i j : Fin (HAdd.hAdd s.length 1)
      hn : Ne i j
      he : Eq (s.toFun i) (s.toFun j)
      this : ∀ (i j : Fin (HAdd.hAdd s.length 1)), Ne i j → Eq (s.toFun i) (s.toFun  …
      hl : Not (LT.lt i j)
      ⊢ False
    -/
  · exact this j i hn.symm he.symm (by omega)
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : Fintype α
    s : LTSeries α
    h : LE.le (Fintype.card α) s.length
    i j : Fin (HAdd.hAdd s.length 1)
    hn : Ne i j
    he : Eq (s.toFun i) (s.toFun j)
    hl : LT.lt i j
    ⊢ False
  -/
  exact absurd he (s.strictMono hl).ne
  /-
    🎉 no goals
  -/


instance [DecidableRel ((· < ·) : α → α → Prop)] : Fintype (LTSeries α) where
  elems := Finset.univ.map (injStrictMono (Fintype.card α))
  complete s := by
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s✝ : Rel β β
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Fintype α
      inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
      s : LTSeries α
      ⊢ Membership.mem (Finset.map (LTSeries.injStrictMono (Fintype.card α)) Finset. …
    -/
    have bl := s.length_lt_card
    /-
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s✝ : Rel β β
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Fintype α
      inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
      s : LTSeries α
      bl : LT.lt s.length (Fintype.card α)
      ⊢ Membership.mem (Finset.map (LTSeries.injStrictMono (Fintype.card α)) Finset. …
    -/
    obtain ⟨l, f, mf⟩ := s
    /-
      case mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Fintype α
      inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
      l : Nat
      f : Fin (HAdd.hAdd l 1) → α
      mf : ∀ (i : Fin l), LT.lt (f i.castSucc) (f i.succ)
      bl : LT.lt { length := l, toFun := f, step := mf }.length (Fintype.card α)
      ⊢ Membership.mem (Finset.map (LTSeries.injStrictMono (Fintype.card α)) Finset. …
    -/
    simp_rw [Finset.mem_map, Finset.mem_univ, true_and, Subtype.exists]
    /-
      case mk
      α : Type u_1
      r : Rel α α
      β : Type u_2
      s : Rel β β
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Fintype α
      inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
      l : Nat
      f : Fin (HAdd.hAdd l 1) → α
      mf : ∀ (i : Fin l), LT.lt (f i.castSucc) (f i.succ)
      bl : LT.lt { length := l, toFun := f, step := mf }.length (Fintype.card α)
      ⊢ Exists fun a => Exists fun b => Eq ((LTSeries.injStrictMono (Fintype.card α) …
    -/
    use ⟨⟨l, bl⟩, f⟩, Fin.strictMono_iff_lt_succ.mpr mf; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma not_finiteDimensionalOrder_iff [Preorder α] [Nonempty α] :
    ¬ FiniteDimensionalOrder α ↔ InfiniteDimensionalOrder α :=
  Rel.not_finiteDimensional_iff


lemma not_infiniteDimensionalOrder_iff [Preorder α] [Nonempty α] :
    ¬ InfiniteDimensionalOrder α ↔ FiniteDimensionalOrder α :=
  Rel.not_infiniteDimensional_iff


variable (α) in
lemma finiteDimensionalOrder_or_infiniteDimensionalOrder [Preorder α] [Nonempty α] :
    FiniteDimensionalOrder α ∨ InfiniteDimensionalOrder α :=
  Rel.finiteDimensional_or_infiniteDimensional _


/-- If `f : α → β` is a strictly monotonic function and `α` is an infinite dimensional type then so
  is `β`. -/
lemma infiniteDimensionalOrder_of_strictMono [Preorder α] [Preorder β]
    (f : α → β) (hf : StrictMono f) [InfiniteDimensionalOrder α] :
    InfiniteDimensionalOrder β :=
  ⟨fun n ↦ ⟨(LTSeries.withLength _ n).map f hf, LTSeries.length_withLength α n⟩⟩

