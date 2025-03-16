/-- If `l` lists all the elements of `α` without duplicates, then `List.get` defines
a bijection `Fin l.length → α`.  See `List.Nodup.getEquivOfForallMemList`
for a version giving an equivalence when there is decidable equality. -/
@[simps]
def getBijectionOfForallMemList (l : List α) (nd : l.Nodup) (h : ∀ x : α, x ∈ l) :
    { f : Fin l.length → α // Function.Bijective f } :=
  ⟨fun i => l.get i, fun _ _ h => nd.get_inj_iff.1 h,
   fun x =>
    let ⟨i, hl⟩ := List.mem_iff_get.1 (h x)
    ⟨i, hl⟩⟩


/-- If `l` has no duplicates, then `List.get` defines an equivalence between `Fin (length l)` and
the set of elements of `l`. -/
@[simps]
def getEquiv (l : List α) (H : Nodup l) : Fin (length l) ≃ { x // x ∈ l } where
  toFun i := ⟨get l i, get_mem _ _⟩
  invFun x := ⟨indexOf (↑x) l, indexOf_lt_length.2 x.2⟩
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     l : List α
                     H : l.Nodup
                     i : Fin l.length
                     ⊢ Eq ((fun x => ⟨List.indexOf (↑x) l, ⋯⟩) ((fun i => ⟨l.get i, ⋯⟩) i)) i
                   -/
  left_inv i := by simp only [List.get_indexOf, eq_self_iff_true, Fin.eta, Subtype.coe_mk, H]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      inst✝ : DecidableEq α
                      l : List α
                      H : l.Nodup
                      x : Subtype fun x => Membership.mem l x
                      ⊢ Eq ((fun i => ⟨l.get i, ⋯⟩) ((fun x => ⟨List.indexOf (↑x) l, ⋯⟩) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- If `l` lists all the elements of `α` without duplicates, then `List.get` defines
an equivalence between `Fin l.length` and `α`.

See `List.Nodup.getBijectionOfForallMemList` for a version without
decidable equality. -/
@[simps]
def getEquivOfForallMemList (l : List α) (nd : l.Nodup) (h : ∀ x : α, x ∈ l) :
    Fin l.length ≃ α where
  toFun i := l.get i
  invFun a := ⟨_, indexOf_lt_length.2 (h a)⟩
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     l : List α
                     nd : l.Nodup
                     h : ∀ (x : α), Membership.mem l x
                     i : Fin l.length
                     ⊢ Eq ((fun a => ⟨List.indexOf a l, ⋯⟩) ((fun i => l.get i) i)) i
                   -/
  left_inv i := by simp [List.indexOf_getElem, nd]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      inst✝ : DecidableEq α
                      l : List α
                      nd : l.Nodup
                      h : ∀ (x : α), Membership.mem l x
                      a : α
                      ⊢ Eq ((fun i => l.get i) ((fun a => ⟨List.indexOf a l, ⋯⟩) a)) a
                    -/
  right_inv a := by simp
                    /-
                      🎉 no goals
                    -/


theorem get_mono (h : l.Sorted (· ≤ ·)) : Monotone l.get := fun _ _ => h.rel_get_of_le


theorem get_strictMono (h : l.Sorted (· < ·)) : StrictMono l.get := fun _ _ => h.rel_get_of_lt


/-- If `l` is a list sorted w.r.t. `(<)`, then `List.get` defines an order isomorphism between
`Fin (length l)` and the set of elements of `l`. -/
def getIso (l : List α) (H : Sorted (· < ·) l) : Fin (length l) ≃o { x // x ∈ l } where
  toEquiv := H.nodup.getEquiv l
  map_rel_iff' := H.get_strictMono.le_iff_le


@[simp]
theorem coe_getIso_apply : (H.getIso l i : α) = get l i :=
  rfl


@[simp]
theorem coe_getIso_symm_apply : ((H.getIso l).symm x : ℕ) = indexOf (↑x) l :=
  rfl


/-- If there is `f`, an order-preserving embedding of `ℕ` into `ℕ` such that
any element of `l` found at index `ix` can be found at index `f ix` in `l'`,
then `Sublist l l'`.
-/
theorem sublist_of_orderEmbedding_get?_eq {l l' : List α} (f : ℕ ↪o ℕ)
    (hf : ∀ ix : ℕ, l.get? ix = l'.get? (f ix)) : l <+ l' := by
  /-
    α : Type u_1
    l l' : List α
    f : OrderEmbedding Nat Nat
    hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
    ⊢ l.Sublist l'
  -/
  induction' l with hd tl IH generalizing l' f
    /-
      case nil
      α : Type u_1
      l' : List α
      f : OrderEmbedding Nat Nat
      hf : ∀ (ix : Nat), Eq (List.nil.get? ix) (l'.get? (f ix))
      ⊢ List.nil.Sublist l'
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    hd : α
    tl : List α
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (tl.get?  …
    l' : List α
    f : OrderEmbedding Nat Nat
    hf : ∀ (ix : Nat), Eq ((List.cons hd tl).get? ix) (l'.get? (f ix))
    ⊢ (List.cons hd tl).Sublist l'
  -/
  have : some hd = _ := hf 0
  /-
    case cons
    α : Type u_1
    hd : α
    tl : List α
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (tl.get?  …
    l' : List α
    f : OrderEmbedding Nat Nat
    hf : ∀ (ix : Nat), Eq ((List.cons hd tl).get? ix) (l'.get? (f ix))
    this : Eq (Option.some hd) (l'.get? (f 0))
    ⊢ (List.cons hd tl).Sublist l'
  -/
  rw [eq_comm, List.get?_eq_some_iff] at this
  /-
    case cons
    α : Type u_1
    hd : α
    tl : List α
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (tl.get?  …
    l' : List α
    f : OrderEmbedding Nat Nat
    hf : ∀ (ix : Nat), Eq ((List.cons hd tl).get? ix) (l'.get? (f ix))
    this : Exists fun h => Eq (l'.get ⟨f 0, h⟩) hd
    ⊢ (List.cons hd tl).Sublist l'
  -/
  obtain ⟨w, h⟩ := this
  let f' : ℕ ↪o ℕ :=
    OrderEmbedding.ofMapLEIff (fun i => f (i + 1) - (f 0 + 1)) fun a b => by
      dsimp only
      rw [Nat.sub_le_sub_iff_right, OrderEmbedding.le_iff_le, Nat.succ_le_succ_iff]
      rw [Nat.succ_le_iff, OrderEmbedding.lt_iff_lt]
      exact b.succ_pos
  /-
    case cons.intro
    α : Type u_1
    hd : α
    tl : List α
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (tl.get?  …
    l' : List α
    f : OrderEmbedding Nat Nat
    hf : ∀ (ix : Nat), Eq ((List.cons hd tl).get? ix) (l'.get? (f ix))
    w : LT.lt (f 0) l'.length
    h : Eq (l'.get ⟨f 0, w⟩) hd
    f' : OrderEmbedding Nat Nat := OrderEmbedding.ofMapLEIff (fun i => HSub.hSub ( …
    ⊢ (List.cons hd tl).Sublist l'
  -/
  simp only [get_eq_getElem] at h
  /-
    case cons.intro
    α : Type u_1
    hd : α
    tl : List α
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (tl.get?  …
    l' : List α
    f : OrderEmbedding Nat Nat
    hf : ∀ (ix : Nat), Eq ((List.cons hd tl).get? ix) (l'.get? (f ix))
    w : LT.lt (f 0) l'.length
    h : Eq (GetElem.getElem l' (f 0) ⋯) hd
    f' : OrderEmbedding Nat Nat := OrderEmbedding.ofMapLEIff (fun i => HSub.hSub ( …
    ⊢ (List.cons hd tl).Sublist l'
  -/
  simp only [get?_eq_getElem?] at hf IH
  have : ∀ ix, tl[ix]? = (l'.drop (f 0 + 1))[f' ix]? := by
    intro ix
    rw [List.getElem?_drop, OrderEmbedding.coe_ofMapLEIff, Nat.add_sub_cancel', ← hf]
    simp only [getElem?_cons_succ]
    rw [Nat.succ_le_iff, OrderEmbedding.lt_iff_lt]
    exact ix.succ_pos
  /-
    case cons.intro
    α : Type u_1
    hd : α
    tl l' : List α
    f : OrderEmbedding Nat Nat
    w : LT.lt (f 0) l'.length
    h : Eq (GetElem.getElem l' (f 0) ⋯) hd
    f' : OrderEmbedding Nat Nat := OrderEmbedding.ofMapLEIff (fun i => HSub.hSub ( …
    hf : ∀ (ix : Nat), Eq (GetElem?.getElem? (List.cons hd tl) ix) (GetElem?.getEl …
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (GetElem? …
    this : ∀ (ix : Nat), Eq (GetElem?.getElem? tl ix) (GetElem?.getElem? (List.dro …
    ⊢ (List.cons hd tl).Sublist l'
  -/
  rw [← List.take_append_drop (f 0 + 1) l', ← List.singleton_append]
  /-
    case cons.intro
    α : Type u_1
    hd : α
    tl l' : List α
    f : OrderEmbedding Nat Nat
    w : LT.lt (f 0) l'.length
    h : Eq (GetElem.getElem l' (f 0) ⋯) hd
    f' : OrderEmbedding Nat Nat := OrderEmbedding.ofMapLEIff (fun i => HSub.hSub ( …
    hf : ∀ (ix : Nat), Eq (GetElem?.getElem? (List.cons hd tl) ix) (GetElem?.getEl …
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (GetElem? …
    this : ∀ (ix : Nat), Eq (GetElem?.getElem? tl ix) (GetElem?.getElem? (List.dro …
    ⊢ (HAppend.hAppend (List.cons hd List.nil) tl).Sublist (HAppend.hAppend (List. …
  -/
  apply List.Sublist.append _ (IH _ this)
  /-
    α : Type u_1
    hd : α
    tl l' : List α
    f : OrderEmbedding Nat Nat
    w : LT.lt (f 0) l'.length
    h : Eq (GetElem.getElem l' (f 0) ⋯) hd
    f' : OrderEmbedding Nat Nat := OrderEmbedding.ofMapLEIff (fun i => HSub.hSub ( …
    hf : ∀ (ix : Nat), Eq (GetElem?.getElem? (List.cons hd tl) ix) (GetElem?.getEl …
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (GetElem? …
    this : ∀ (ix : Nat), Eq (GetElem?.getElem? tl ix) (GetElem?.getElem? (List.dro …
    ⊢ (List.cons hd List.nil).Sublist (List.take (HAdd.hAdd (f 0) 1) l')
  -/
  rw [List.singleton_sublist, ← h, l'.getElem_take' _ (Nat.lt_succ_self _)]
  /-
    α : Type u_1
    hd : α
    tl l' : List α
    f : OrderEmbedding Nat Nat
    w : LT.lt (f 0) l'.length
    h : Eq (GetElem.getElem l' (f 0) ⋯) hd
    f' : OrderEmbedding Nat Nat := OrderEmbedding.ofMapLEIff (fun i => HSub.hSub ( …
    hf : ∀ (ix : Nat), Eq (GetElem?.getElem? (List.cons hd tl) ix) (GetElem?.getEl …
    IH : ∀ {l' : List α} (f : OrderEmbedding Nat Nat), (∀ (ix : Nat), Eq (GetElem? …
    this : ∀ (ix : Nat), Eq (GetElem?.getElem? tl ix) (GetElem?.getElem? (List.dro …
    ⊢ Membership.mem (List.take (HAdd.hAdd (f 0) 1) l') (GetElem.getElem (List.tak …
  -/
  exact List.getElem_mem _
  /-
    🎉 no goals
  -/


/-- A `l : List α` is `Sublist l l'` for `l' : List α` iff
there is `f`, an order-preserving embedding of `ℕ` into `ℕ` such that
any element of `l` found at index `ix` can be found at index `f ix` in `l'`.
-/
theorem sublist_iff_exists_orderEmbedding_get?_eq {l l' : List α} :
    l <+ l' ↔ ∃ f : ℕ ↪o ℕ, ∀ ix : ℕ, l.get? ix = l'.get? (f ix) := by
  /-
    α : Type u_1
    l l' : List α
    ⊢ Iff (l.Sublist l') (Exists fun f => ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      l l' : List α
      ⊢ l.Sublist l' → Exists fun f => ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
    -/
  · intro H
    induction H with
    | slnil => simp
    | cons _ _ IH =>
      obtain ⟨f, hf⟩ := IH
      refine ⟨f.trans (OrderEmbedding.ofStrictMono (· + 1) fun _ => by simp), ?_⟩
      simpa using hf
    | cons₂ _ _ IH =>
      obtain ⟨f, hf⟩ := IH
      refine
        ⟨OrderEmbedding.ofMapLEIff (fun ix : ℕ => if ix = 0 then 0 else (f ix.pred).succ) ?_, ?_⟩
      · rintro ⟨_ | a⟩ ⟨_ | b⟩ <;> simp [Nat.succ_le_succ_iff]
      · rintro ⟨_ | i⟩
        · simp
        · simpa using hf _
    /-
      case mpr
      α : Type u_1
      l l' : List α
      ⊢ (Exists fun f => ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))) → l.Sublist l'
    -/
  · rintro ⟨f, hf⟩
    /-
      case mpr.intro
      α : Type u_1
      l l' : List α
      f : OrderEmbedding Nat Nat
      hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
      ⊢ l.Sublist l'
    -/
    exact sublist_of_orderEmbedding_get?_eq f hf
    /-
      🎉 no goals
    -/


/-- A `l : List α` is `Sublist l l'` for `l' : List α` iff
there is `f`, an order-preserving embedding of `Fin l.length` into `Fin l'.length` such that
any element of `l` found at index `ix` can be found at index `f ix` in `l'`.
-/
theorem sublist_iff_exists_fin_orderEmbedding_get_eq {l l' : List α} :
    l <+ l' ↔
      ∃ f : Fin l.length ↪o Fin l'.length,
        ∀ ix : Fin l.length, l.get ix = l'.get (f ix) := by
  /-
    α : Type u_1
    l l' : List α
    ⊢ Iff (l.Sublist l') (Exists fun f => ∀ (ix : Fin l.length), Eq (l.get ix) (l' …
  -/
  rw [sublist_iff_exists_orderEmbedding_get?_eq]
  /-
    α : Type u_1
    l l' : List α
    ⊢ Iff (Exists fun f => ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))) (Exists  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      l l' : List α
      ⊢ (Exists fun f => ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))) → Exists fun …
    -/
  · rintro ⟨f, hf⟩
    have h : ∀ {i : ℕ}, i < l.length → f i < l'.length := by
      intro i hi
      specialize hf i
      rw [get?_eq_get hi, eq_comm, get?_eq_some_iff] at hf
      obtain ⟨h, -⟩ := hf
      exact h
    /-
      case mp.intro
      α : Type u_1
      l l' : List α
      f : OrderEmbedding Nat Nat
      hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
      h : ∀ {i : Nat}, LT.lt i l.length → LT.lt (f i) l'.length
      ⊢ Exists fun f => ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
    -/
    refine ⟨OrderEmbedding.ofMapLEIff (fun ix => ⟨f ix, h ix.is_lt⟩) ?_, ?_⟩
      /-
        case mp.intro.refine_1
        α : Type u_1
        l l' : List α
        f : OrderEmbedding Nat Nat
        hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
        h : ∀ {i : Nat}, LT.lt i l.length → LT.lt (f i) l'.length
        ⊢ ∀ (a b : Fin l.length), Iff (LE.le ((fun ix => ⟨f ↑ix, ⋯⟩) a) ((fun ix => ⟨f …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.refine_2
        α : Type u_1
        l l' : List α
        f : OrderEmbedding Nat Nat
        hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
        h : ∀ {i : Nat}, LT.lt i l.length → LT.lt (f i) l'.length
        ⊢ ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get ((OrderEmbedding.ofMapLEIff (fu …
      -/
    · intro i
      /-
        case mp.intro.refine_2
        α : Type u_1
        l l' : List α
        f : OrderEmbedding Nat Nat
        hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
        h : ∀ {i : Nat}, LT.lt i l.length → LT.lt (f i) l'.length
        i : Fin l.length
        ⊢ Eq (l.get i) (l'.get ((OrderEmbedding.ofMapLEIff (fun ix => ⟨f ↑ix, ⋯⟩) ⋯) i))
      -/
      apply Option.some_injective
      /-
        case mp.intro.refine_2.a
        α : Type u_1
        l l' : List α
        f : OrderEmbedding Nat Nat
        hf : ∀ (ix : Nat), Eq (l.get? ix) (l'.get? (f ix))
        h : ∀ {i : Nat}, LT.lt i l.length → LT.lt (f i) l'.length
        i : Fin l.length
        ⊢ Eq (Option.some (l.get i)) (Option.some (l'.get ((OrderEmbedding.ofMapLEIff  …
      -/
      simpa [getElem?_eq_getElem i.2, getElem?_eq_getElem (h i.2)] using hf i
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      l l' : List α
      ⊢ (Exists fun f => ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))) → Exi …
    -/
  · rintro ⟨f, hf⟩
    refine
      ⟨OrderEmbedding.ofStrictMono (fun i => if hi : i < l.length then f ⟨i, hi⟩ else i + l'.length)
          ?_,
        ?_⟩
      /-
        case mpr.intro.refine_1
        α : Type u_1
        l l' : List α
        f : OrderEmbedding (Fin l.length) (Fin l'.length)
        hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
        ⊢ StrictMono fun i => dite (LT.lt i l.length) (fun hi => ↑(f ⟨i, hi⟩)) fun hi  …
      -/
    · intro i j h
      /-
        case mpr.intro.refine_1
        α : Type u_1
        l l' : List α
        f : OrderEmbedding (Fin l.length) (Fin l'.length)
        hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
        i j : Nat
        h : LT.lt i j
        ⊢ LT.lt ((fun i => dite (LT.lt i l.length) (fun hi => ↑(f ⟨i, hi⟩)) fun hi =>  …
      -/
      dsimp only
      /-
        case mpr.intro.refine_1
        α : Type u_1
        l l' : List α
        f : OrderEmbedding (Fin l.length) (Fin l'.length)
        hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
        i j : Nat
        h : LT.lt i j
        ⊢ LT.lt (dite (LT.lt i l.length) (fun hi => ↑(f ⟨i, hi⟩)) fun hi => HAdd.hAdd  …
      -/
      split_ifs with hi hj hj
        /-
          case pos
          α : Type u_1
          l l' : List α
          f : OrderEmbedding (Fin l.length) (Fin l'.length)
          hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
          i j : Nat
          h : LT.lt i j
          hi : LT.lt i l.length
          hj : LT.lt j l.length
          ⊢ LT.lt ↑(f ⟨i, hi⟩) ↑(f ⟨j, hj⟩)
        -/
      · rwa [Fin.val_fin_lt, f.lt_iff_lt]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          l l' : List α
          f : OrderEmbedding (Fin l.length) (Fin l'.length)
          hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
          i j : Nat
          h : LT.lt i j
          hi : LT.lt i l.length
          hj : Not (LT.lt j l.length)
          ⊢ LT.lt (↑(f ⟨i, hi⟩)) (HAdd.hAdd j l'.length)
        -/
      · omega
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          l l' : List α
          f : OrderEmbedding (Fin l.length) (Fin l'.length)
          hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
          i j : Nat
          h : LT.lt i j
          hi : Not (LT.lt i l.length)
          hj : LT.lt j l.length
          ⊢ LT.lt (HAdd.hAdd i l'.length) ↑(f ⟨j, hj⟩)
        -/
      · exact absurd (h.trans hj) hi
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          l l' : List α
          f : OrderEmbedding (Fin l.length) (Fin l'.length)
          hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
          i j : Nat
          h : LT.lt i j
          hi : Not (LT.lt i l.length)
          hj : Not (LT.lt j l.length)
          ⊢ LT.lt (HAdd.hAdd i l'.length) (HAdd.hAdd j l'.length)
        -/
      · simpa using h
        /-
          🎉 no goals
        -/
      /-
        case mpr.intro.refine_2
        α : Type u_1
        l l' : List α
        f : OrderEmbedding (Fin l.length) (Fin l'.length)
        hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
        ⊢ ∀ (ix : Nat), Eq (l.get? ix) (l'.get? ((OrderEmbedding.ofStrictMono (fun i = …
      -/
    · intro i
      /-
        case mpr.intro.refine_2
        α : Type u_1
        l l' : List α
        f : OrderEmbedding (Fin l.length) (Fin l'.length)
        hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
        i : Nat
        ⊢ Eq (l.get? i) (l'.get? ((OrderEmbedding.ofStrictMono (fun i => dite (LT.lt i …
      -/
      simp only [OrderEmbedding.coe_ofStrictMono]
      /-
        case mpr.intro.refine_2
        α : Type u_1
        l l' : List α
        f : OrderEmbedding (Fin l.length) (Fin l'.length)
        hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
        i : Nat
        ⊢ Eq (l.get? i) (l'.get? (dite (LT.lt i l.length) (fun hi => ↑(f ⟨i, hi⟩)) fun …
      -/
      split_ifs with hi
        /-
          case pos
          α : Type u_1
          l l' : List α
          f : OrderEmbedding (Fin l.length) (Fin l'.length)
          hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
          i : Nat
          hi : LT.lt i l.length
          ⊢ Eq (l.get? i) (l'.get? ↑(f ⟨i, hi⟩))
        -/
      · rw [get?_eq_get hi, get?_eq_get, ← hf]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          l l' : List α
          f : OrderEmbedding (Fin l.length) (Fin l'.length)
          hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
          i : Nat
          hi : Not (LT.lt i l.length)
          ⊢ Eq (l.get? i) (l'.get? (HAdd.hAdd i l'.length))
        -/
      · rw [get?_eq_none_iff.mpr, get?_eq_none_iff.mpr]
          /-
            case neg
            α : Type u_1
            l l' : List α
            f : OrderEmbedding (Fin l.length) (Fin l'.length)
            hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
            i : Nat
            hi : Not (LT.lt i l.length)
            ⊢ LE.le l'.length (HAdd.hAdd i l'.length)
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case neg
            α : Type u_1
            l l' : List α
            f : OrderEmbedding (Fin l.length) (Fin l'.length)
            hf : ∀ (ix : Fin l.length), Eq (l.get ix) (l'.get (f ix))
            i : Nat
            hi : Not (LT.lt i l.length)
            ⊢ LE.le l.length i
          -/
        · simpa using hi
          /-
            🎉 no goals
          -/


/-- An element `x : α` of `l : List α` is a duplicate iff it can be found
at two distinct indices `n m : ℕ` inside the list `l`.
-/
theorem duplicate_iff_exists_distinct_get {l : List α} {x : α} :
    l.Duplicate x ↔
      ∃ (n m : Fin l.length) (_ : n < m),
        x = l.get n ∧ x = l.get m := by
  classical
    rw [duplicate_iff_two_le_count, le_count_iff_replicate_sublist,
      sublist_iff_exists_fin_orderEmbedding_get_eq]
    constructor
    · rintro ⟨f, hf⟩
      refine ⟨f ⟨0, by simp⟩, f ⟨1, by simp⟩, f.lt_iff_lt.2 (Nat.zero_lt_one), ?_⟩
      rw [← hf, ← hf]; simp
    · rintro ⟨n, m, hnm, h, h'⟩
      refine ⟨OrderEmbedding.ofStrictMono (fun i => if (i : ℕ) = 0 then n else m) ?_, ?_⟩
      · rintro ⟨⟨_ | i⟩, hi⟩ ⟨⟨_ | j⟩, hj⟩
        · simp
        · simp [hnm]
        · simp
        · simp only [Nat.lt_succ_iff, Nat.succ_le_succ_iff, replicate, length, Nat.le_zero] at hi hj
          simp [hi, hj]
      · rintro ⟨⟨_ | i⟩, hi⟩
        · simpa using h
        · simpa using h'


