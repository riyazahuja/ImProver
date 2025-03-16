instance instLinearOrder : LinearOrder (Fin n) :=
  @LinearOrder.liftWithOrd (Fin n) _ _ ⟨fun x y => ⟨max x y, max_rec' (· < n) x.2 y.2⟩⟩
    ⟨fun x y => ⟨min x y, min_rec' (· < n) x.2 y.2⟩⟩ _ Fin.val Fin.val_injective (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)


instance instBoundedOrder [NeZero n] : BoundedOrder (Fin n) where
  top := rev 0
  le_top i := Nat.le_pred_of_lt i.is_lt
  bot := 0
  bot_le := Fin.zero_le'

/- There is a slight asymmetry here, in the sense that `0` is of type `Fin n` when we have
`[NeZero n]` whereas `last n` is of type `Fin (n + 1)`. To address this properly would
require a change to std4, defining `NeZero n` and thus re-defining `last n`
(and possibly make its argument implicit) as `rev 0`, of type `Fin n`. As we can see from these
lemmas, this would be equivalent to the existing definition. -/


instance instPartialOrder : PartialOrder (Fin n) := inferInstance

instance instLattice      : Lattice (Fin n)      := inferInstance


lemma top_eq_last (n : ℕ) : ⊤ = Fin.last n := rfl


lemma bot_eq_zero (n : ℕ) : ⊥ = (0 : Fin (n + 1)) := rfl


@[simp] theorem rev_bot [NeZero n] : rev (⊥ : Fin n) = ⊤ := rfl

@[simp] theorem rev_top [NeZero n] : rev (⊤ : Fin n) = ⊥ := rev_rev _


theorem rev_zero_eq_top (n : ℕ) [NeZero n] : rev (0 : Fin n) = ⊤ := rfl

                                                         /-
                                                           n : Nat
                                                           ⊢ Eq (Fin.last n).rev Bot.bot
                                                         -/
theorem rev_last_eq_bot (n : ℕ) : rev (last n) = ⊥ := by rw [rev_last, bot_eq_zero]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma strictMono_pred_comp (hf : ∀ a, f a ≠ 0) (hf₂ : StrictMono f) :
    StrictMono (fun a => pred (f a) (hf a)) := fun _ _ h => pred_lt_pred_iff.2 (hf₂ h)


lemma monotone_pred_comp (hf : ∀ a, f a ≠ 0) (hf₂ : Monotone f) :
    Monotone (fun a => pred (f a) (hf a)) := fun _ _ h => pred_le_pred_iff.2 (hf₂ h)


lemma strictMono_castPred_comp (hf : ∀ a, f a ≠ last n) (hf₂ : StrictMono f) :
    StrictMono (fun a => castPred (f a) (hf a)) := fun _ _ h => castPred_lt_castPred_iff.2 (hf₂ h)


lemma monotone_castPred_comp (hf : ∀ a, f a ≠ last n) (hf₂ : Monotone f) :
    Monotone (fun a => castPred (f a) (hf a)) := fun _ _ h => castPred_le_castPred_iff.2 (hf₂ h)


/-- A function `f` on `Fin (n + 1)` is strictly monotone if and only if `f i < f (i + 1)`
for all `i`. -/
lemma strictMono_iff_lt_succ : StrictMono f ↔ ∀ i : Fin n, f (castSucc i) < f i.succ :=
  liftFun_iff_succ (· < ·)


/-- A function `f` on `Fin (n + 1)` is monotone if and only if `f i ≤ f (i + 1)` for all `i`. -/
lemma monotone_iff_le_succ : Monotone f ↔ ∀ i : Fin n, f (castSucc i) ≤ f i.succ :=
  monotone_iff_forall_lt.trans <| liftFun_iff_succ (· ≤ ·)


/-- A function `f` on `Fin (n + 1)` is strictly antitone if and only if `f (i + 1) < f i`
for all `i`. -/
lemma strictAnti_iff_succ_lt : StrictAnti f ↔ ∀ i : Fin n, f i.succ < f (castSucc i) :=
  liftFun_iff_succ (· > ·)


/-- A function `f` on `Fin (n + 1)` is antitone if and only if `f (i + 1) ≤ f i` for all `i`. -/
lemma antitone_iff_succ_le : Antitone f ↔ ∀ i : Fin n, f i.succ ≤ f (castSucc i) :=
  antitone_iff_forall_lt.trans <| liftFun_iff_succ (· ≥ ·)


lemma val_strictMono : StrictMono (val : Fin n → ℕ) := fun _ _ ↦ id

lemma cast_strictMono {k l : ℕ} (h : k = l) : StrictMono (cast h) := fun {_ _} h ↦ h


lemma strictMono_succ : StrictMono (succ : Fin n → Fin (n + 1)) := fun _ _ ↦ succ_lt_succ

lemma strictMono_castLE (h : n ≤ m) : StrictMono (castLE h : Fin n → Fin m) := fun _ _ ↦ id

lemma strictMono_castAdd (m) : StrictMono (castAdd m : Fin n → Fin (n + m)) := strictMono_castLE _

lemma strictMono_castSucc : StrictMono (castSucc : Fin n → Fin (n + 1)) := strictMono_castAdd _

lemma strictMono_natAdd (n) : StrictMono (natAdd n : Fin m → Fin (n + m)) :=
  fun i j h ↦ Nat.add_lt_add_left (show i.val < j.val from h) _

lemma strictMono_addNat (m) : StrictMono ((addNat · m) : Fin n → Fin (n + m)) :=
  fun i j h ↦ Nat.add_lt_add_right (show i.val < j.val from h) _


lemma strictMono_succAbove (p : Fin (n + 1)) : StrictMono (succAbove p) :=
  strictMono_castSucc.ite strictMono_succ
    (fun _ _ hij hj => (castSucc_lt_castSucc_iff.mpr hij).trans hj) fun i =>
    (castSucc_lt_succ i).le


lemma succAbove_lt_succAbove_iff : succAbove p i < succAbove p j ↔ i < j :=
  (strictMono_succAbove p).lt_iff_lt


lemma succAbove_le_succAbove_iff : succAbove p i ≤ succAbove p j ↔ i ≤ j :=
  (strictMono_succAbove p).le_iff_le


lemma predAbove_right_monotone (p : Fin n) : Monotone p.predAbove := fun a b H => by
  /-
    n : Nat
    p : Fin n
    a b : Fin (HAdd.hAdd n 1)
    H : LE.le a b
    ⊢ LE.le (p.predAbove a) (p.predAbove b)
  -/
  dsimp [predAbove]
  /-
    n : Nat
    p : Fin n
    a b : Fin (HAdd.hAdd n 1)
    H : LE.le a b
    ⊢ LE.le (dite (LT.lt p.castSucc a) (fun h => a.pred ⋯) fun h => a.castPred ⋯)  …
  -/
  split_ifs with ha hb hb
  /-
    case pos
    n : Nat
    p : Fin n
    a b : Fin (HAdd.hAdd n 1)
    H : LE.le a b
    ha : LT.lt p.castSucc a
    hb : LT.lt p.castSucc b
    ⊢ LE.le (a.pred ⋯) (b.pred ⋯)
  -/
  all_goals simp only [le_iff_val_le_val, coe_pred]
    /-
      case pos
      n : Nat
      p : Fin n
      a b : Fin (HAdd.hAdd n 1)
      H : LE.le a b
      ha : LT.lt p.castSucc a
      hb : LT.lt p.castSucc b
      ⊢ LE.le (HSub.hSub (↑a) 1) (HSub.hSub (↑b) 1)
    -/
  · exact pred_le_pred H
    /-
      🎉 no goals
    -/
  · calc
      _ ≤ _ := Nat.pred_le _
      _ ≤ _ := H
    /-
      case pos
      n : Nat
      p : Fin n
      a b : Fin (HAdd.hAdd n 1)
      H : LE.le a b
      ha : Not (LT.lt p.castSucc a)
      hb : LT.lt p.castSucc b
      ⊢ LE.le (↑(a.castPred ⋯)) (HSub.hSub (↑b) 1)
    -/
  · exact le_pred_of_lt ((not_lt.mp ha).trans_lt hb)
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      p : Fin n
      a b : Fin (HAdd.hAdd n 1)
      H : LE.le a b
      ha : Not (LT.lt p.castSucc a)
      hb : Not (LT.lt p.castSucc b)
      ⊢ LE.le ↑(a.castPred ⋯) ↑(b.castPred ⋯)
    -/
  · exact H
    /-
      🎉 no goals
    -/


lemma predAbove_left_monotone (i : Fin (n + 1)) : Monotone fun p ↦ predAbove p i := fun a b H ↦ by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    a b : Fin n
    H : LE.le a b
    ⊢ LE.le ((fun p => p.predAbove i) a) ((fun p => p.predAbove i) b)
  -/
  dsimp [predAbove]
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    a b : Fin n
    H : LE.le a b
    ⊢ LE.le (dite (LT.lt a.castSucc i) (fun h => i.pred ⋯) fun h => i.castPred ⋯)  …
  -/
  split_ifs with ha hb hb
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a b : Fin n
      H : LE.le a b
      ha : LT.lt a.castSucc i
      hb : LT.lt b.castSucc i
      ⊢ LE.le (i.pred ⋯) (i.pred ⋯)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a b : Fin n
      H : LE.le a b
      ha : LT.lt a.castSucc i
      hb : Not (LT.lt b.castSucc i)
      ⊢ LE.le (i.pred ⋯) (i.castPred ⋯)
    -/
  · exact pred_le _
    /-
      🎉 no goals
    -/
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a b : Fin n
      H : LE.le a b
      ha : Not (LT.lt a.castSucc i)
      hb : LT.lt b.castSucc i
      ⊢ LE.le (i.castPred ⋯) (i.pred ⋯)
    -/
  · have : b < a := castSucc_lt_castSucc_iff.mpr (hb.trans_le (le_of_not_gt ha))
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a b : Fin n
      H : LE.le a b
      ha : Not (LT.lt a.castSucc i)
      hb : LT.lt b.castSucc i
      this : LT.lt b a
      ⊢ LE.le (i.castPred ⋯) (i.pred ⋯)
    -/
    exact absurd H this.not_le
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a b : Fin n
      H : LE.le a b
      ha : Not (LT.lt a.castSucc i)
      hb : Not (LT.lt b.castSucc i)
      ⊢ LE.le (i.castPred ⋯) (i.castPred ⋯)
    -/
  · rfl
    /-
      🎉 no goals
    -/


/--  `Fin.predAbove p` as an `OrderHom`. -/
@[simps!] def predAboveOrderHom (p : Fin n) : Fin (n + 1) →o Fin n :=
  ⟨p.predAbove, p.predAbove_right_monotone⟩


/-- The equivalence `Fin n ≃ {i // i < n}` is an order isomorphism. -/
@[simps! apply symm_apply]
def orderIsoSubtype : Fin n ≃o {i // i < n} :=
                              /-
                                m n : Nat
                                p : Fin (HAdd.hAdd n 1)
                                i j : Fin n
                                ⊢ Monotone ⇑Fin.equivSubtype
                              -/
                              /-
                                🎉 no goals
                              -/
  equivSubtype.toOrderIso (by simp [Monotone]) (by simp [Monotone])
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `Fin.cast` as an `OrderIso`.

`castOrderIso eq i` embeds `i` into an equal `Fin` type. -/
@[simps]
def castOrderIso (eq : n = m) : Fin n ≃o Fin m where
  toEquiv := ⟨cast eq, cast eq.symm, leftInverse_cast eq, rightInverse_cast eq⟩
  map_rel_iff' := cast_le_cast eq


@[deprecated (since := "2024-05-23")] alias castIso := castOrderIso


@[simp]
                                                                                        /-
                                                                                          m n : Nat
                                                                                          h : Eq n m
                                                                                          ⊢ Eq (Fin.castOrderIso h).symm (Fin.castOrderIso ⋯)
                                                                                        -/
lemma symm_castOrderIso (h : n = m) : (castOrderIso h).symm = castOrderIso h.symm := by subst h; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


@[deprecated (since := "2024-05-23")] alias symm_castIso := symm_castOrderIso


@[simp]
                                                                                          /-
                                                                                            n : Nat
                                                                                            h : optParam (Eq n n) ⋯
                                                                                            ⊢ Eq (Fin.castOrderIso h) (OrderIso.refl (Fin n))
                                                                                          -/
lemma castOrderIso_refl (h : n = n := rfl) : castOrderIso h = OrderIso.refl (Fin n) := by ext; simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[deprecated (since := "2024-05-23")] alias castIso_refl := castOrderIso_refl


/-- While in many cases `Fin.castOrderIso` is better than `Equiv.cast`/`cast`, sometimes we want to
apply a generic lemma about `cast`. -/
lemma castOrderIso_toEquiv (h : n = m) : (castOrderIso h).toEquiv = Equiv.cast (h ▸ rfl) := by
  /-
    m n : Nat
    h : Eq n m
    ⊢ Eq (Fin.castOrderIso h).toEquiv (Equiv.cast ⋯)
  -/
  subst h; rfl
           /-
             🎉 no goals
           -/


@[deprecated (since := "2024-05-23")] alias castIso_to_equiv := castOrderIso_toEquiv


/-- `Fin.rev n` as an order-reversing isomorphism. -/
@[simps! apply toEquiv]
def revOrderIso : (Fin n)ᵒᵈ ≃o Fin n := ⟨OrderDual.ofDual.trans revPerm, rev_le_rev⟩


@[simp]
lemma revOrderIso_symm_apply (i : Fin n) : revOrderIso.symm i = OrderDual.toDual (rev i) := rfl


/-- The inclusion map `Fin n → ℕ` is an order embedding. -/
@[simps! apply]
def valOrderEmb (n) : Fin n ↪o ℕ := ⟨valEmbedding, Iff.rfl⟩


/-- The ordering on `Fin n` is a well order. -/
instance Lt.isWellOrder (n) : IsWellOrder (Fin n) (· < ·) := (valOrderEmb n).isWellOrder


/-- `Fin.succ` as an `OrderEmbedding` -/
def succOrderEmb (n : ℕ) : Fin n ↪o Fin (n + 1) := .ofStrictMono succ strictMono_succ


@[simp, norm_cast] lemma coe_succOrderEmb : ⇑(succOrderEmb n) = Fin.succ := rfl


/-- `Fin.castLE` as an `OrderEmbedding`.

`castLEEmb h i` embeds `i` into a larger `Fin` type. -/
@[simps! apply toEmbedding]
def castLEOrderEmb (h : n ≤ m) : Fin n ↪o Fin m := .ofStrictMono (castLE h) (strictMono_castLE h)


/-- `Fin.castAdd` as an `OrderEmbedding`.

`castAddEmb m i` embeds `i : Fin n` in `Fin (n+m)`. See also `Fin.natAddEmb` and `Fin.addNatEmb`. -/
@[simps! apply toEmbedding]
def castAddOrderEmb (m) : Fin n ↪o Fin (n + m) := .ofStrictMono (castAdd m) (strictMono_castAdd m)


/-- `Fin.castSucc` as an `OrderEmbedding`.

`castSuccOrderEmb i` embeds `i : Fin n` in `Fin (n+1)`. -/
@[simps! apply toEmbedding]
def castSuccOrderEmb : Fin n ↪o Fin (n + 1) := .ofStrictMono castSucc strictMono_castSucc


/-- `Fin.addNat` as an `OrderEmbedding`.

`addNatOrderEmb m i` adds `m` to `i`, generalizes `Fin.succ`. -/
@[simps! apply toEmbedding]
def addNatOrderEmb (m) : Fin n ↪o Fin (n + m) := .ofStrictMono (addNat · m) (strictMono_addNat m)


/-- `Fin.natAdd` as an `OrderEmbedding`.

`natAddOrderEmb n i` adds `n` to `i` "on the left". -/
@[simps! apply toEmbedding]
def natAddOrderEmb (n) : Fin m ↪o Fin (n + m) := .ofStrictMono (natAdd n) (strictMono_natAdd n)


/--  `Fin.succAbove p` as an `OrderEmbedding`. -/
@[simps! apply toEmbedding]
def succAboveOrderEmb (p : Fin (n + 1)) : Fin n ↪o Fin (n + 1) :=
  OrderEmbedding.ofStrictMono (succAbove p) (strictMono_succAbove p)


/-- If `e` is an `orderIso` between `Fin n` and `Fin m`, then `n = m` and `e` is the identity
map. In this lemma we state that for each `i : Fin n` we have `(e i : ℕ) = (i : ℕ)`. -/
@[simp] lemma coe_orderIso_apply (e : Fin n ≃o Fin m) (i : Fin n) : (e i : ℕ) = i := by
  /-
    m n : Nat
    e : OrderIso (Fin n) (Fin m)
    i : Fin n
    ⊢ Eq ↑(e i) ↑i
  -/
  rcases i with ⟨i, hi⟩
  /-
    case mk
    m n : Nat
    e : OrderIso (Fin n) (Fin m)
    i : Nat
    hi : LT.lt i n
    ⊢ Eq ↑(e ⟨i, hi⟩) ↑⟨i, hi⟩
  -/
  dsimp only
  /-
    case mk
    m n : Nat
    e : OrderIso (Fin n) (Fin m)
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (↑(e ⟨i, hi⟩)) i
  -/
  induction' i using Nat.strong_induction_on with i h
  /-
    case mk.h
    m n : Nat
    e : OrderIso (Fin n) (Fin m)
    i : Nat
    h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
    hi : LT.lt i n
    ⊢ Eq (↑(e ⟨i, hi⟩)) i
  -/
  refine le_antisymm (forall_lt_iff_le.1 fun j hj => ?_) (forall_lt_iff_le.1 fun j hj => ?_)
    /-
      case mk.h.refine_1
      m n : Nat
      e : OrderIso (Fin n) (Fin m)
      i : Nat
      h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j ↑(e ⟨i, hi⟩)
      ⊢ LT.lt j i
    -/
  · have := e.symm.lt_iff_lt.2 (mk_lt_of_lt_val hj)
    /-
      case mk.h.refine_1
      m n : Nat
      e : OrderIso (Fin n) (Fin m)
      i : Nat
      h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j ↑(e ⟨i, hi⟩)
      this : LT.lt (e.symm ⟨j, ⋯⟩) (e.symm (e ⟨i, hi⟩))
      ⊢ LT.lt j i
    -/
    rw [e.symm_apply_apply] at this
    -- Porting note: convert was abusing definitional equality
    /-
      case mk.h.refine_1
      m n : Nat
      e : OrderIso (Fin n) (Fin m)
      i : Nat
      h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j ↑(e ⟨i, hi⟩)
      this : LT.lt (e.symm ⟨j, ⋯⟩) ⟨i, hi⟩
      ⊢ LT.lt j i
    -/
    have : _ < i := this
    /-
      case mk.h.refine_1
      m n : Nat
      e : OrderIso (Fin n) (Fin m)
      i : Nat
      h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j ↑(e ⟨i, hi⟩)
      this✝ : LT.lt (e.symm ⟨j, ⋯⟩) ⟨i, hi⟩
      this : LT.lt (↑(e.symm ⟨j, ⋯⟩)) i
      ⊢ LT.lt j i
    -/
    convert this
    /-
      case h.e'_3
      m n : Nat
      e : OrderIso (Fin n) (Fin m)
      i : Nat
      h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j ↑(e ⟨i, hi⟩)
      this✝ : LT.lt (e.symm ⟨j, ⋯⟩) ⟨i, hi⟩
      this : LT.lt (↑(e.symm ⟨j, ⋯⟩)) i
      ⊢ Eq j ↑(e.symm ⟨j, ⋯⟩)
    -/
    simpa using h _ this (e.symm _).is_lt
    /-
      🎉 no goals
    -/
    /-
      case mk.h.refine_2
      m n : Nat
      e : OrderIso (Fin n) (Fin m)
      i : Nat
      h : ∀ (m_1 : Nat), LT.lt m_1 i → ∀ (hi : LT.lt m_1 n), Eq (↑(e ⟨m_1, hi⟩)) m_1
      hi : LT.lt i n
      j : Nat
      hj : LT.lt j i
      ⊢ LT.lt j ↑(e ⟨i, hi⟩)
    -/
  · rwa [← h j hj (hj.trans hi), ← lt_iff_val_lt_val, e.lt_iff_lt]
    /-
      🎉 no goals
    -/


/-- Two strictly monotone functions from `Fin n` are equal provided that their ranges
are equal. -/
@[deprecated StrictMono.range_inj (since := "2024-09-17")]
lemma strictMono_unique {f g : Fin n → α} (hf : StrictMono f) (hg : StrictMono g)
    (h : range f = range g) : f = g :=
  (hf.range_inj hg).1 h


/-- Two order embeddings of `Fin n` are equal provided that their ranges are equal. -/
@[deprecated OrderEmbedding.range_inj (since := "2024-09-17")]
lemma orderEmbedding_eq {f g : Fin n ↪o α} (h : range f = range g) : f = g :=
  OrderEmbedding.range_inj.1 h


