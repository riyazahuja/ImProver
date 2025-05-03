/-- The maximal reduction of a word. It is computable
iff `α` has decidable equality. -/
@[to_additive "The maximal reduction of a word. It is computable iff `α` has decidable equality."]
def reduce : (L : List (α × Bool)) -> List (α × Bool) :=
  List.rec [] fun hd1 _tl1 ih =>
    List.casesOn ih [hd1] fun hd2 tl2 =>
      if hd1.1 = hd2.1 ∧ hd1.2 = not hd2.2 then tl2 else hd1 :: hd2 :: tl2


@[to_additive (attr := simp)] lemma reduce_nil : reduce ([] : List (α × Bool)) = [] := rfl

@[to_additive] lemma reduce_singleton (s : α × Bool) : reduce [s] = [s] := rfl


@[to_additive (attr := simp)]
theorem reduce.cons (x) :
    reduce (x :: L) =
      List.casesOn (reduce L) [x] fun hd tl =>
        if x.1 = hd.1 ∧ x.2 = not hd.2 then tl else x :: hd :: tl :=
  rfl


@[to_additive (attr := simp)]
theorem reduce_replicate (n : ℕ) (x : α × Bool) :
    reduce (.replicate n x) = .replicate n x := by
  induction n with
  | zero => simp [reduce]
  | succ n ih =>
    rw [List.replicate_succ, reduce.cons, ih]
    cases n with
    | zero => simp
    | succ n => simp [List.replicate_succ]


/-- The first theorem that characterises the function `reduce`: a word reduces to its maximal
  reduction. -/
@[to_additive "The first theorem that characterises the function `reduce`: a word reduces to its
  maximal reduction."]
theorem reduce.red : Red L (reduce L) := by
  induction L with
  | nil => constructor
  | cons hd1 tl1 ih =>
    dsimp
    revert ih
    generalize htl : reduce tl1 = TL
    intro ih
    cases TL with
    | nil => exact Red.cons_cons ih
    | cons hd2 tl2 =>
      dsimp only
      split_ifs with h
      · cases hd1
        cases hd2
        cases h
        dsimp at *
        subst_vars
        apply Red.trans (Red.cons_cons ih)
        exact Red.Step.cons_not_rev.to_red
      · exact Red.cons_cons ih


@[to_additive]
theorem reduce.not {p : Prop} :
    ∀ {L₁ L₂ L₃ : List (α × Bool)} {x b}, reduce L₁ = L₂ ++ (x, b) :: (x, !b) :: L₃ → p
                                    /-
                                      α : Type u_1
                                      inst✝ : DecidableEq α
                                      p : Prop
                                      L2 L3 : List (Prod α Bool)
                                      x✝¹ : α
                                      x✝ : Bool
                                      h : Eq (FreeGroup.reduce List.nil) (HAppend.hAppend L2 (List.cons { fst := x✝¹ …
                                      ⊢ p
                                    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  | [], L2, L3, _, _ => fun h => by cases L2 <;> injections
                                                 /-
                                                   🎉 no goals
                                                 -/
  | (x, b) :: L1, L2, L3, x', b' => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      p : Prop
      x : α
      b : Bool
      L1 L2 L3 : List (Prod α Bool)
      x' : α
      b' : Bool
      ⊢ Eq (FreeGroup.reduce (List.cons { fst := x, snd := b } L1)) (HAppend.hAppend …
    -/
    dsimp
    cases r : reduce L1 with
    | nil =>
      dsimp; intro h
      exfalso
      have := congr_arg List.length h
      simp? [List.length] at this says
        simp only [List.length, zero_add, List.length_append] at this
      omega
    | cons hd tail =>
      cases' hd with y c
      dsimp only
      split_ifs with h <;> intro H
      · rw [H] at r
        exact @reduce.not _ L1 ((y, c) :: L2) L3 x' b' r
      rcases L2 with (_ | ⟨a, L2⟩)
      · injections; subst_vars
        simp at h
      · refine @reduce.not _ L1 L2 L3 x' b' ?_
        injection H with _ H
        rw [r, H]; rfl


/-- The second theorem that characterises the function `reduce`: the maximal reduction of a word
only reduces to itself. -/
@[to_additive "The second theorem that characterises the function `reduce`: the maximal reduction of
  a word only reduces to itself."]
theorem reduce.min (H : Red (reduce L₁) L₂) : reduce L₁ = L₂ := by
  /-
    α : Type u_1
    L₁ L₂ : List (Prod α Bool)
    inst✝ : DecidableEq α
    H : FreeGroup.Red (FreeGroup.reduce L₁) L₂
    ⊢ Eq (FreeGroup.reduce L₁) L₂
  -/
  induction' H with L1 L' L2 H1 H2 ih
    /-
      case refl
      α : Type u_1
      L₁ L₂ : List (Prod α Bool)
      inst✝ : DecidableEq α
      ⊢ Eq (FreeGroup.reduce L₁) (FreeGroup.reduce L₁)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case tail
      α : Type u_1
      L₁ L₂ : List (Prod α Bool)
      inst✝ : DecidableEq α
      L1 L' : List (Prod α Bool)
      L2 : Relation.ReflTransGen FreeGroup.Red.Step (FreeGroup.reduce L₁) L1
      H1 : FreeGroup.Red.Step L1 L'
      H2 : Eq (FreeGroup.reduce L₁) L1
      ⊢ Eq (FreeGroup.reduce L₁) L'
    -/
  · cases' H1 with L4 L5 x b
    /-
      case tail.not
      α : Type u_1
      L₁ L₂ : List (Prod α Bool)
      inst✝ : DecidableEq α
      L4 L5 : List (Prod α Bool)
      x : α
      b : Bool
      L2 : Relation.ReflTransGen FreeGroup.Red.Step (FreeGroup.reduce L₁) (HAppend.h …
      H2 : Eq (FreeGroup.reduce L₁) (HAppend.hAppend L4 (List.cons { fst := x, snd : …
      ⊢ Eq (FreeGroup.reduce L₁) (HAppend.hAppend L4 L5)
    -/
    exact reduce.not H2
    /-
      🎉 no goals
    -/


/-- `reduce` is idempotent, i.e. the maximal reduction of the maximal reduction of a word is the
  maximal reduction of the word. -/
@[to_additive (attr := simp) "`reduce` is idempotent, i.e. the maximal reduction of the maximal
  reduction of a word is the maximal reduction of the word."]
theorem reduce.idem : reduce (reduce L) = reduce L :=
  Eq.symm <| reduce.min reduce.red


@[to_additive]
theorem reduce.Step.eq (H : Red.Step L₁ L₂) : reduce L₁ = reduce L₂ :=
  let ⟨_L₃, HR13, HR23⟩ := Red.church_rosser reduce.red (reduce.red.head H)
  (reduce.min HR13).trans (reduce.min HR23).symm


/-- If a word reduces to another word, then they have a common maximal reduction. -/
@[to_additive "If a word reduces to another word, then they have a common maximal reduction."]
theorem reduce.eq_of_red (H : Red L₁ L₂) : reduce L₁ = reduce L₂ :=
  let ⟨_L₃, HR13, HR23⟩ := Red.church_rosser reduce.red (Red.trans H reduce.red)
  (reduce.min HR13).trans (reduce.min HR23).symm


alias red.reduce_eq := reduce.eq_of_red


alias freeAddGroup.red.reduce_eq := FreeAddGroup.reduce.eq_of_red


@[to_additive]
theorem Red.reduce_right (h : Red L₁ L₂) : Red L₁ (reduce L₂) :=
  reduce.eq_of_red h ▸ reduce.red


@[to_additive]
theorem Red.reduce_left (h : Red L₁ L₂) : Red L₂ (reduce L₁) :=
  (reduce.eq_of_red h).symm ▸ reduce.red


/-- If two words correspond to the same element in the free group, then they
have a common maximal reduction. This is the proof that the function that sends
an element of the free group to its maximal reduction is well-defined. -/
@[to_additive "If two words correspond to the same element in the additive free group, then they
  have a common maximal reduction. This is the proof that the function that sends an element of the
  free group to its maximal reduction is well-defined."]
theorem reduce.sound (H : mk L₁ = mk L₂) : reduce L₁ = reduce L₂ :=
  let ⟨_L₃, H13, H23⟩ := Red.exact.1 H
  (reduce.eq_of_red H13).trans (reduce.eq_of_red H23).symm


/-- If two words have a common maximal reduction, then they correspond to the same element in the
  free group. -/
@[to_additive "If two words have a common maximal reduction, then they correspond to the same
  element in the additive free group."]
theorem reduce.exact (H : reduce L₁ = reduce L₂) : mk L₁ = mk L₂ :=
  Red.exact.2 ⟨reduce L₂, H ▸ reduce.red, reduce.red⟩


/-- A word and its maximal reduction correspond to the same element of the free group. -/
@[to_additive "A word and its maximal reduction correspond to the same element of the additive free
  group."]
theorem reduce.self : mk (reduce L) = mk L :=
  reduce.exact reduce.idem


/-- If words `w₁ w₂` are such that `w₁` reduces to `w₂`, then `w₂` reduces to the maximal reduction
  of `w₁`. -/
@[to_additive "If words `w₁ w₂` are such that `w₁` reduces to `w₂`, then `w₂` reduces to the maximal
  reduction of `w₁`."]
theorem reduce.rev (H : Red L₁ L₂) : Red L₂ (reduce L₁) :=
  (reduce.eq_of_red H).symm ▸ reduce.red


/-- The function that sends an element of the free group to its maximal reduction. -/
@[to_additive "The function that sends an element of the additive free group to its maximal
  reduction."]
def toWord : FreeGroup α → List (α × Bool) :=
  Quot.lift reduce fun _L₁ _L₂ H => reduce.Step.eq H


@[to_additive]
                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : DecidableEq α
                                                                   ⊢ ∀ {x : FreeGroup α}, Eq (FreeGroup.mk x.toWord) x
                                                                 -/
theorem mk_toWord : ∀ {x : FreeGroup α}, mk (toWord x) = x := by rintro ⟨L⟩; exact reduce.self
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[to_additive]
theorem toWord_injective : Function.Injective (toWord : FreeGroup α → List (α × Bool)) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ Function.Injective FreeGroup.toWord
  -/
  rintro ⟨L₁⟩ ⟨L₂⟩; exact reduce.exact
                    /-
                      🎉 no goals
                    -/


@[to_additive (attr := simp)]
theorem toWord_inj {x y : FreeGroup α} : toWord x = toWord y ↔ x = y :=
  toWord_injective.eq_iff


@[to_additive (attr := simp)]
theorem toWord_mk : (mk L₁).toWord = reduce L₁ :=
  rfl


@[to_additive (attr := simp)]
theorem toWord_of (a : α) : (of a).toWord = [(a, true)] :=
  rfl


@[to_additive (attr := simp)]
theorem reduce_toWord : ∀ x : FreeGroup α, reduce (toWord x) = toWord x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ ∀ (x : FreeGroup α), Eq (FreeGroup.reduce x.toWord) x.toWord
  -/
  rintro ⟨L⟩
  /-
    case mk
    α : Type u_1
    inst✝ : DecidableEq α
    x✝ : FreeGroup α
    L : List (Prod α Bool)
    ⊢ Eq (FreeGroup.reduce (FreeGroup.toWord (Quot.mk FreeGroup.Red.Step L))) (Fre …
  -/
  exact reduce.idem
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem toWord_one : (1 : FreeGroup α).toWord = [] :=
  rfl


@[to_additive (attr := simp)]
theorem toWord_of_pow (a : α) (n : ℕ) : (of a ^ n).toWord = List.replicate n (a, true) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (HPow.hPow (FreeGroup.of a) n).toWord (List.replicate n { fst := a, snd : …
  -/
  rw [of, pow_mk, List.flatten_replicate_singleton, toWord]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (Quot.lift FreeGroup.reduce ⋯ (FreeGroup.mk (List.replicate n { fst := a, …
  -/
  exact reduce_replicate _ _
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem toWord_eq_nil_iff {x : FreeGroup α} : x.toWord = [] ↔ x = 1 :=
  toWord_injective.eq_iff' toWord_one


@[to_additive]
theorem reduce_invRev {w : List (α × Bool)} : reduce (invRev w) = invRev (reduce w) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    w : List (Prod α Bool)
    ⊢ Eq (FreeGroup.reduce (FreeGroup.invRev w)) (FreeGroup.invRev (FreeGroup.redu …
  -/
  apply reduce.min
  /-
    case H
    α : Type u_1
    inst✝ : DecidableEq α
    w : List (Prod α Bool)
    ⊢ FreeGroup.Red (FreeGroup.reduce (FreeGroup.invRev w)) (FreeGroup.invRev (Fre …
  -/
  rw [← red_invRev_iff, invRev_invRev]
  /-
    case H
    α : Type u_1
    inst✝ : DecidableEq α
    w : List (Prod α Bool)
    ⊢ FreeGroup.Red (FreeGroup.invRev (FreeGroup.reduce (FreeGroup.invRev w))) (Fr …
  -/
  apply Red.reduce_left
  /-
    case H.h
    α : Type u_1
    inst✝ : DecidableEq α
    w : List (Prod α Bool)
    ⊢ FreeGroup.Red w (FreeGroup.invRev (FreeGroup.reduce (FreeGroup.invRev w)))
  -/
  have : Red (invRev (invRev w)) (invRev (reduce (invRev w))) := reduce.red.invRev
  /-
    case H.h
    α : Type u_1
    inst✝ : DecidableEq α
    w : List (Prod α Bool)
    this : FreeGroup.Red (FreeGroup.invRev (FreeGroup.invRev w)) (FreeGroup.invRev …
    ⊢ FreeGroup.Red w (FreeGroup.invRev (FreeGroup.reduce (FreeGroup.invRev w)))
  -/
  rwa [invRev_invRev] at this
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem toWord_inv (x : FreeGroup α) : x⁻¹.toWord = invRev x.toWord := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : FreeGroup α
    ⊢ Eq (Inv.inv x).toWord (FreeGroup.invRev x.toWord)
  -/
  rcases x with ⟨L⟩
  /-
    case mk
    α : Type u_1
    inst✝ : DecidableEq α
    x : FreeGroup α
    L : List (Prod α Bool)
    ⊢ Eq (Inv.inv (Quot.mk FreeGroup.Red.Step L)).toWord (FreeGroup.invRev (FreeGr …
  -/
  rw [quot_mk_eq_mk, inv_mk, toWord_mk, toWord_mk, reduce_invRev]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma toWord_mul_sublist (x y : FreeGroup α) : (x * y).toWord <+ x.toWord ++ y.toWord := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : FreeGroup α
    ⊢ (HMul.hMul x y).toWord.Sublist (HAppend.hAppend x.toWord y.toWord)
  -/
  refine Red.sublist ?_
  have : x * y = FreeGroup.mk (x.toWord ++ y.toWord) := by
    rw [← FreeGroup.mul_mk, FreeGroup.mk_toWord, FreeGroup.mk_toWord]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : FreeGroup α
    this : Eq (HMul.hMul x y) (FreeGroup.mk (HAppend.hAppend x.toWord y.toWord))
    ⊢ FreeGroup.Red (HAppend.hAppend x.toWord y.toWord) (HMul.hMul x y).toWord
  -/
  rw [this]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : FreeGroup α
    this : Eq (HMul.hMul x y) (FreeGroup.mk (HAppend.hAppend x.toWord y.toWord))
    ⊢ FreeGroup.Red (HAppend.hAppend x.toWord y.toWord) (FreeGroup.mk (HAppend.hAp …
  -/
  exact FreeGroup.reduce.red
  /-
    🎉 no goals
  -/


/-- **Constructive Church-Rosser theorem** (compare `church_rosser`). -/
@[to_additive "**Constructive Church-Rosser theorem** (compare `church_rosser`)."]
def reduce.churchRosser (H12 : Red L₁ L₂) (H13 : Red L₁ L₃) : { L₄ // Red L₂ L₄ ∧ Red L₃ L₄ } :=
  ⟨reduce L₁, reduce.rev H12, reduce.rev H13⟩


@[to_additive]
instance : DecidableEq (FreeGroup α) :=
  toWord_injective.decidableEq

-- TODO @[to_additive] doesn't succeed, possibly due to a bug

instance Red.decidableRel : DecidableRel (@Red α)
  | [], [] => isTrue Red.refl
  | [], _hd2 :: _tl2 => isFalse fun H => List.noConfusion (Red.nil_iff.1 H)
  | (x, b) :: tl, [] =>
    match Red.decidableRel tl [(x, not b)] with
    | isTrue H => isTrue <| Red.trans (Red.cons_cons H) <| (@Red.Step.not _ [] [] _ _).to_red
    | isFalse H => isFalse fun H2 => H <| Red.cons_nil_iff_singleton.1 H2
  | (x1, b1) :: tl1, (x2, b2) :: tl2 =>
    if h : (x1, b1) = (x2, b2) then
      match Red.decidableRel tl1 tl2 with
      | isTrue H => isTrue <| h ▸ Red.cons_cons H
      | isFalse H => isFalse fun H2 => H <| (Red.cons_cons_iff _).1 <| h.symm ▸ H2
    else
      match Red.decidableRel tl1 ((x1, ! b1) :: (x2, b2) :: tl2) with
      | isTrue H => isTrue <| (Red.cons_cons H).tail Red.Step.cons_not
      | isFalse H => isFalse fun H2 => H <| Red.inv_of_red_of_ne h H2


/-- A list containing every word that `w₁` reduces to. -/
def Red.enum (L₁ : List (α × Bool)) : List (List (α × Bool)) :=
  List.filter (Red L₁) (List.sublists L₁)


theorem Red.enum.sound (H : L₂ ∈ List.filter (Red L₁) (List.sublists L₁)) : Red L₁ L₂ :=
  of_decide_eq_true (@List.of_mem_filter _ _ L₂ _ H)


theorem Red.enum.complete (H : Red L₁ L₂) : L₂ ∈ Red.enum L₁ :=
  List.mem_filter_of_mem (List.mem_sublists.2 <| Red.sublist H) (decide_eq_true H)


instance (L₁ : List (α × Bool)) : Fintype { L₂ // Red L₁ L₂ } :=
  Fintype.subtype (List.toFinset <| Red.enum L₁) fun _L₂ =>
    ⟨fun H => Red.enum.sound <| List.mem_toFinset.1 H, fun H =>
      List.mem_toFinset.2 <| Red.enum.complete H⟩


@[to_additive (attr := simp)]
theorem one_ne_of (a : α) : 1 ≠ of a :=
                                                         /-
                                                           α : Type u_1
                                                           a : α
                                                           this : DecidableEq α := Classical.decEq α
                                                           ⊢ Ne (FreeGroup.toWord 1) (FreeGroup.of a).toWord
                                                         -/
  letI := Classical.decEq α; ne_of_apply_ne toWord <| by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive (attr := simp)]
theorem of_ne_one (a : α) : of a ≠ 1 := one_ne_of _ |>.symm


@[to_additive]
instance [Nonempty α] : Nontrivial (FreeGroup α) where
  exists_pair_ne := let ⟨x⟩ := ‹Nonempty α›; ⟨1, of x, one_ne_of x⟩


/-- The length of reduced words provides a norm on a free group. -/
@[to_additive "The length of reduced words provides a norm on an additive free group."]
def norm (x : FreeGroup α) : ℕ :=
  x.toWord.length


@[to_additive (attr := simp)]
theorem norm_inv_eq {x : FreeGroup α} : norm x⁻¹ = norm x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : FreeGroup α
    ⊢ Eq (Inv.inv x).norm x.norm
  -/
  simp only [norm, toWord_inv, invRev_length]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem norm_eq_zero {x : FreeGroup α} : norm x = 0 ↔ x = 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x : FreeGroup α
    ⊢ Iff (Eq x.norm 0) (Eq x 1)
  -/
  simp only [norm, List.length_eq_zero, toWord_eq_nil_iff]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem norm_one : norm (1 : FreeGroup α) = 0 :=
  rfl


@[to_additive (attr := simp)]
theorem norm_of (a : α) : norm (of a) = 1 :=
  rfl


@[to_additive]
theorem norm_mk_le : norm (mk L₁) ≤ L₁.length :=
  reduce.red.length_le


@[to_additive]
theorem norm_mul_le (x y : FreeGroup α) : norm (x * y) ≤ norm x + norm y :=
  calc
                                                          /-
                                                            α : Type u_1
                                                            inst✝ : DecidableEq α
                                                            x y : FreeGroup α
                                                            ⊢ Eq (HMul.hMul x y).norm (FreeGroup.mk (HAppend.hAppend x.toWord y.toWord)).n …
                                                          -/
    norm (x * y) = norm (mk (x.toWord ++ y.toWord)) := by rw [← mul_mk, mk_toWord, mk_toWord]
                                                          /-
                                                            🎉 no goals
                                                          -/
    _ ≤ (x.toWord ++ y.toWord).length := norm_mk_le
    _ = norm x + norm y := List.length_append _ _


@[to_additive (attr := simp)]
theorem norm_of_pow (a : α) (n : ℕ) : norm (of a ^ n) = n := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (HPow.hPow (FreeGroup.of a) n).norm n
  -/
  rw [norm, toWord_of_pow, List.length_replicate]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem norm_surjective [Nonempty α] : Function.Surjective (norm (α := α)) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    ⊢ Function.Surjective FreeGroup.norm
  -/
  let ⟨a⟩ := ‹Nonempty α›
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    a : α
    ⊢ Function.Surjective FreeGroup.norm
  -/
  exact Function.RightInverse.surjective <| norm_of_pow a
  /-
    🎉 no goals
  -/


