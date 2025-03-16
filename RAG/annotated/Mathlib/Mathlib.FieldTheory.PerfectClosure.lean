/-- `PerfectClosure.R` is the relation `(n, x) ∼ (n + 1, x ^ p)` for `n : ℕ` and `x : K`.
`PerfectClosure K p` is the quotient by this relation. -/
@[mk_iff]
inductive PerfectClosure.R : ℕ × K → ℕ × K → Prop
  | intro : ∀ n x, PerfectClosure.R (n, x) (n + 1, frobenius K p x)


/-- The perfect closure is the smallest extension that makes frobenius surjective. -/
def PerfectClosure : Type u :=
  Quot (PerfectClosure.R K p)


/-- `PerfectClosure.mk K p (n, x)` for `n : ℕ` and `x : K` is an element of `PerfectClosure K p`,
viewed as `x ^ (p ^ -n)`. Every element of `PerfectClosure K p` is of this form
(`PerfectClosure.mk_surjective`). -/
def mk (x : ℕ × K) : PerfectClosure K p :=
  Quot.mk (R K p) x


theorem mk_surjective : Function.Surjective (mk K p) := Quot.mk_surjective


@[simp] theorem mk_succ_pow (m : ℕ) (x : K) : mk K p ⟨m + 1, x ^ p⟩ = mk K p ⟨m, x⟩ :=
  Eq.symm <| Quot.sound (R.intro m x)


@[simp]
theorem quot_mk_eq_mk (x : ℕ × K) : (Quot.mk (R K p) x : PerfectClosure K p) = mk K p x :=
  rfl


/-- Lift a function `ℕ × K → L` to a function on `PerfectClosure K p`. -/
def liftOn {L : Type*} (x : PerfectClosure K p) (f : ℕ × K → L)
    (hf : ∀ x y, R K p x y → f x = f y) : L :=
  Quot.liftOn x f hf


@[simp]
theorem liftOn_mk {L : Sort _} (f : ℕ × K → L) (hf : ∀ x y, R K p x y → f x = f y) (x : ℕ × K) :
    (mk K p x).liftOn f hf = f x :=
  rfl


@[elab_as_elim]
theorem induction_on (x : PerfectClosure K p) {q : PerfectClosure K p → Prop}
    (h : ∀ x, q (mk K p x)) : q x :=
  Quot.inductionOn x h


private theorem mul_aux_left (x1 x2 y : ℕ × K) (H : R K p x1 x2) :
    mk K p (x1.1 + y.1, (frobenius K p)^[y.1] x1.2 * (frobenius K p)^[x1.1] y.2) =
      mk K p (x2.1 + y.1, (frobenius K p)^[y.1] x2.2 * (frobenius K p)^[x2.1] y.2) :=
  match x1, x2, H with
  | _, _, R.intro n x =>
    Quot.sound <| by
      rw [← iterate_succ_apply, iterate_succ_apply', iterate_succ_apply', ← frobenius_mul,
        Nat.succ_add]
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        x1 x2 y : Prod Nat K
        H : PerfectClosure.R K p x1 x2
        n : Nat
        x : K
        ⊢ PerfectClosure.R K p { fst := HAdd.hAdd { fst := n, snd := x }.1 y.1, snd := …
      -/
      apply R.intro
      /-
        🎉 no goals
      -/


private theorem mul_aux_right (x y1 y2 : ℕ × K) (H : R K p y1 y2) :
    mk K p (x.1 + y1.1, (frobenius K p)^[y1.1] x.2 * (frobenius K p)^[x.1] y1.2) =
      mk K p (x.1 + y2.1, (frobenius K p)^[y2.1] x.2 * (frobenius K p)^[x.1] y2.2) :=
  match y1, y2, H with
  | _, _, R.intro n y =>
    Quot.sound <| by
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        x y1 y2 : Prod Nat K
        H : PerfectClosure.R K p y1 y2
        n : Nat
        y : K
        ⊢ PerfectClosure.R K p { fst := HAdd.hAdd x.1 { fst := n, snd := y }.1, snd := …
      -/
      rw [← iterate_succ_apply, iterate_succ_apply', iterate_succ_apply', ← frobenius_mul]
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        x y1 y2 : Prod Nat K
        H : PerfectClosure.R K p y1 y2
        n : Nat
        y : K
        ⊢ PerfectClosure.R K p { fst := HAdd.hAdd x.1 { fst := n, snd := y }.1, snd := …
      -/
      apply R.intro
      /-
        🎉 no goals
      -/


instance instMul : Mul (PerfectClosure K p) :=
  ⟨Quot.lift
      (fun x : ℕ × K =>
        Quot.lift
          (fun y : ℕ × K =>
            mk K p (x.1 + y.1, (frobenius K p)^[y.1] x.2 * (frobenius K p)^[x.1] y.2))
          (mul_aux_right K p x))
      fun x1 x2 (H : R K p x1 x2) =>
      funext fun e => Quot.inductionOn e fun y => mul_aux_left K p x1 x2 y H⟩


@[simp]
theorem mk_mul_mk (x y : ℕ × K) :
    mk K p x * mk K p y =
      mk K p (x.1 + y.1, (frobenius K p)^[y.1] x.2 * (frobenius K p)^[x.1] y.2) :=
  rfl


instance instCommMonoid : CommMonoid (PerfectClosure K p) :=
  { (inferInstance : Mul (PerfectClosure K p)) with
    mul_assoc := fun e f g =>
      Quot.inductionOn e fun ⟨m, x⟩ =>
        Quot.inductionOn f fun ⟨n, y⟩ =>
          Quot.inductionOn g fun ⟨s, z⟩ => by
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk (PerfectClosure.R K p) { fst := m, snd :=  …
            -/
            simp only [quot_mk_eq_mk, mk_mul_mk] -- Porting note: added this line
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd (HAdd.hAdd m n) s, snd := HMul. …
            -/
            apply congr_arg (Quot.mk _)
            simp only [add_assoc, mul_assoc, iterate_map_mul, ← iterate_add_apply,
              add_comm, add_left_comm]
    one := mk K p (0, 1)
    one_mul := fun e =>
      Quot.inductionOn e fun ⟨n, x⟩ =>
        congr_arg (Quot.mk _) <| by
          /-
            K : Type u
            inst✝² : CommRing K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            e : PerfectClosure K p
            x✝ : Prod Nat K
            n : Nat
            x : K
            ⊢ Eq { fst := HAdd.hAdd { fst := 0, snd := 1 }.1 { fst := n, snd := x }.1, snd …
          -/
          simp only [iterate_map_one, iterate_zero_apply, one_mul, zero_add]
          /-
            🎉 no goals
          -/
    mul_one := fun e =>
      Quot.inductionOn e fun ⟨n, x⟩ =>
        congr_arg (Quot.mk _) <| by
          /-
            K : Type u
            inst✝² : CommRing K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            e : PerfectClosure K p
            x✝ : Prod Nat K
            n : Nat
            x : K
            ⊢ Eq { fst := HAdd.hAdd { fst := n, snd := x }.1 { fst := 0, snd := 1 }.1, snd …
          -/
          simp only [iterate_map_one, iterate_zero_apply, mul_one, add_zero]
          /-
            🎉 no goals
          -/
    mul_comm := fun e f =>
      Quot.inductionOn e fun ⟨m, x⟩ =>
        Quot.inductionOn f fun ⟨n, y⟩ =>
                                      /-
                                        K : Type u
                                        inst✝² : CommRing K
                                        p : Nat
                                        inst✝¹ : Fact (Nat.Prime p)
                                        inst✝ : CharP K p
                                        e f : PerfectClosure K p
                                        x✝¹ : Prod Nat K
                                        m : Nat
                                        x : K
                                        x✝ : Prod Nat K
                                        n : Nat
                                        y : K
                                        ⊢ Eq { fst := HAdd.hAdd { fst := m, snd := x }.1 { fst := n, snd := y }.1, snd …
                                      -/
          congr_arg (Quot.mk _) <| by simp only [add_comm, mul_comm] }
                                      /-
                                        🎉 no goals
                                      -/


theorem one_def : (1 : PerfectClosure K p) = mk K p (0, 1) :=
  rfl


instance instInhabited : Inhabited (PerfectClosure K p) :=
  ⟨1⟩


private theorem add_aux_left (x1 x2 y : ℕ × K) (H : R K p x1 x2) :
    mk K p (x1.1 + y.1, (frobenius K p)^[y.1] x1.2 + (frobenius K p)^[x1.1] y.2) =
      mk K p (x2.1 + y.1, (frobenius K p)^[y.1] x2.2 + (frobenius K p)^[x2.1] y.2) :=
  match x1, x2, H with
  | _, _, R.intro n x =>
    Quot.sound <| by
      rw [← iterate_succ_apply, iterate_succ_apply', iterate_succ_apply', ← frobenius_add,
        Nat.succ_add]
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        x1 x2 y : Prod Nat K
        H : PerfectClosure.R K p x1 x2
        n : Nat
        x : K
        ⊢ PerfectClosure.R K p { fst := HAdd.hAdd { fst := n, snd := x }.1 y.1, snd := …
      -/
      apply R.intro
      /-
        🎉 no goals
      -/


private theorem add_aux_right (x y1 y2 : ℕ × K) (H : R K p y1 y2) :
    mk K p (x.1 + y1.1, (frobenius K p)^[y1.1] x.2 + (frobenius K p)^[x.1] y1.2) =
      mk K p (x.1 + y2.1, (frobenius K p)^[y2.1] x.2 + (frobenius K p)^[x.1] y2.2) :=
  match y1, y2, H with
  | _, _, R.intro n y =>
    Quot.sound <| by
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        x y1 y2 : Prod Nat K
        H : PerfectClosure.R K p y1 y2
        n : Nat
        y : K
        ⊢ PerfectClosure.R K p { fst := HAdd.hAdd x.1 { fst := n, snd := y }.1, snd := …
      -/
      rw [← iterate_succ_apply, iterate_succ_apply', iterate_succ_apply', ← frobenius_add]
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        x y1 y2 : Prod Nat K
        H : PerfectClosure.R K p y1 y2
        n : Nat
        y : K
        ⊢ PerfectClosure.R K p { fst := HAdd.hAdd x.1 { fst := n, snd := y }.1, snd := …
      -/
      apply R.intro
      /-
        🎉 no goals
      -/


instance instAdd : Add (PerfectClosure K p) :=
  ⟨Quot.lift
      (fun x : ℕ × K =>
        Quot.lift
          (fun y : ℕ × K =>
            mk K p (x.1 + y.1, (frobenius K p)^[y.1] x.2 + (frobenius K p)^[x.1] y.2))
          (add_aux_right K p x))
      fun x1 x2 (H : R K p x1 x2) =>
      funext fun e => Quot.inductionOn e fun y => add_aux_left K p x1 x2 y H⟩


@[simp]
theorem mk_add_mk (x y : ℕ × K) :
    mk K p x + mk K p y =
      mk K p (x.1 + y.1, (frobenius K p)^[y.1] x.2 + (frobenius K p)^[x.1] y.2) :=
  rfl


instance instNeg : Neg (PerfectClosure K p) :=
  ⟨Quot.lift (fun x : ℕ × K => mk K p (x.1, -x.2)) fun x y (H : R K p x y) =>
      match x, y, H with
                                              /-
                                                K : Type u
                                                inst✝² : CommRing K
                                                p : Nat
                                                inst✝¹ : Fact (Nat.Prime p)
                                                inst✝ : CharP K p
                                                x✝ y : Prod Nat K
                                                H : PerfectClosure.R K p x✝ y
                                                n : Nat
                                                x : K
                                                ⊢ PerfectClosure.R K p { fst := { fst := n, snd := x }.1, snd := Neg.neg { fst …
                                              -/
      | _, _, R.intro n x => Quot.sound <| by rw [← frobenius_neg]; apply R.intro⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem neg_mk (x : ℕ × K) : -mk K p x = mk K p (x.1, -x.2) :=
  rfl


instance instZero : Zero (PerfectClosure K p) :=
  ⟨mk K p (0, 0)⟩


theorem zero_def : (0 : PerfectClosure K p) = mk K p (0, 0) :=
  rfl


/-- Prior to https://github.com/leanprover-community/mathlib4/pull/15862, this lemma was called `mk_zero_zero`.
See `mk_zero_right` for the lemma used to be called `mk_zero`. -/
@[simp]
theorem mk_zero : mk K p 0 = 0 :=
  rfl


@[deprecated (since := "2024-08-16")] alias mk_zero_zero := mk_zero

-- Porting note: improved proof structure

@[simp]
theorem mk_zero_right (n : ℕ) : mk K p (n, 0) = 0 := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    ⊢ Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) 0
  -/
  induction' n with n ih
    /-
      case zero
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      ⊢ Eq (PerfectClosure.mk K p { fst := 0, snd := 0 }) 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) 0
    ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd n 1, snd := 0 }) 0
  -/
  rw [← ih]
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) 0
    ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd n 1, snd := 0 }) (PerfectClosur …
  -/
  symm
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) 0
    ⊢ Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) (PerfectClosure.mk K p { f …
  -/
  apply Quot.sound
  /-
    case succ.a
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) 0
    ⊢ PerfectClosure.R K p { fst := n, snd := 0 } { fst := HAdd.hAdd n 1, snd := 0 }
  -/
  have := R.intro (p := p) n (0 : K)
  /-
    case succ.a
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := 0 }) 0
    this : PerfectClosure.R K p { fst := n, snd := 0 } { fst := HAdd.hAdd n 1, snd …
    ⊢ PerfectClosure.R K p { fst := n, snd := 0 } { fst := HAdd.hAdd n 1, snd := 0 }
  -/
  rwa [frobenius_zero K p] at this
  /-
    🎉 no goals
  -/

-- Porting note: improved proof structure

theorem R.sound (m n : ℕ) (x y : K) (H : (frobenius K p)^[m] x = y) :
    mk K p (n, x) = mk K p (m + n, y) := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    m n : Nat
    x y : K
    H : Eq (Nat.iterate (⇑(frobenius K p)) m x) y
    ⊢ Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p { f …
  -/
  subst H
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    m n : Nat
    x : K
    ⊢ Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p { f …
  -/
  induction' m with m ih
    /-
      case zero
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      n : Nat
      x : K
      ⊢ Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p { f …
    -/
  · simp only [zero_add, iterate_zero_apply]
    /-
      🎉 no goals
    -/
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    x : K
    m : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p  …
    ⊢ Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p { f …
  -/
  rw [ih, Nat.succ_add, iterate_succ']
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    x : K
    m : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p  …
    ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd m n, snd := Nat.iterate (⇑(frob …
  -/
  apply Quot.sound
  /-
    case succ.a
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    x : K
    m : Nat
    ih : Eq (PerfectClosure.mk K p { fst := n, snd := x }) (PerfectClosure.mk K p  …
    ⊢ PerfectClosure.R K p { fst := HAdd.hAdd m n, snd := Nat.iterate (⇑(frobenius …
  -/
  apply R.intro
  /-
    🎉 no goals
  -/


instance instAddCommGroup : AddCommGroup (PerfectClosure K p) :=
  { (inferInstance : Add (PerfectClosure K p)),
    (inferInstance : Neg (PerfectClosure K p)) with
    add_assoc := fun e f g =>
      Quot.inductionOn e fun ⟨m, x⟩ =>
        Quot.inductionOn f fun ⟨n, y⟩ =>
          Quot.inductionOn g fun ⟨s, z⟩ => by
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Quot.mk (PerfectClosure.R K p) { fst := m, snd :=  …
            -/
            simp only [quot_mk_eq_mk, mk_add_mk] -- Porting note: added this line
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd (HAdd.hAdd m n) s, snd := HAdd. …
            -/
            apply congr_arg (Quot.mk _)
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq { fst := HAdd.hAdd (HAdd.hAdd m n) s, snd := HAdd.hAdd (Nat.iterate (⇑(fr …
            -/
            simp only [iterate_map_add, ← iterate_add_apply, add_assoc, add_comm s _]
            /-
              🎉 no goals
            -/
    zero := 0
    zero_add := fun e =>
      Quot.inductionOn e fun ⟨n, x⟩ =>
        congr_arg (Quot.mk _) <| by
          /-
            K : Type u
            inst✝² : CommRing K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            e : PerfectClosure K p
            x✝ : Prod Nat K
            n : Nat
            x : K
            ⊢ Eq { fst := HAdd.hAdd { fst := 0, snd := 0 }.1 { fst := n, snd := x }.1, snd …
          -/
          simp only [iterate_map_zero, iterate_zero_apply, zero_add]
          /-
            🎉 no goals
          -/
    add_zero := fun e =>
      Quot.inductionOn e fun ⟨n, x⟩ =>
        congr_arg (Quot.mk _) <| by
          /-
            K : Type u
            inst✝² : CommRing K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            e : PerfectClosure K p
            x✝ : Prod Nat K
            n : Nat
            x : K
            ⊢ Eq { fst := HAdd.hAdd { fst := n, snd := x }.1 { fst := 0, snd := 0 }.1, snd …
          -/
          simp only [iterate_map_zero, iterate_zero_apply, add_zero]
          /-
            🎉 no goals
          -/
    sub_eq_add_neg := fun _ _ => rfl
    neg_add_cancel := fun e =>
      Quot.inductionOn e fun ⟨n, x⟩ => by
        /-
          K : Type u
          inst✝² : CommRing K
          p : Nat
          inst✝¹ : Fact (Nat.Prime p)
          inst✝ : CharP K p
          e : PerfectClosure K p
          x✝ : Prod Nat K
          n : Nat
          x : K
          ⊢ Eq (HAdd.hAdd (Neg.neg (Quot.mk (PerfectClosure.R K p) { fst := n, snd := x  …
        -/
        simp only [quot_mk_eq_mk, neg_mk, mk_add_mk, iterate_map_neg, neg_add_cancel, mk_zero_right]
        /-
          🎉 no goals
        -/
    add_comm := fun e f =>
      Quot.inductionOn e fun ⟨m, x⟩ =>
                                                                     /-
                                                                       K : Type u
                                                                       inst✝² : CommRing K
                                                                       p : Nat
                                                                       inst✝¹ : Fact (Nat.Prime p)
                                                                       inst✝ : CharP K p
                                                                       e f : PerfectClosure K p
                                                                       x✝¹ : Prod Nat K
                                                                       m : Nat
                                                                       x : K
                                                                       x✝ : Prod Nat K
                                                                       n : Nat
                                                                       y : K
                                                                       ⊢ Eq { fst := HAdd.hAdd { fst := m, snd := x }.1 { fst := n, snd := y }.1, snd …
                                                                     -/
        Quot.inductionOn f fun ⟨n, y⟩ => congr_arg (Quot.mk _) <| by simp only [add_comm]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    nsmul := nsmulRec
    zsmul := zsmulRec }


instance instCommRing : CommRing (PerfectClosure K p) :=
  { instAddCommGroup K p, AddMonoidWithOne.unary,
    (inferInstance : CommMonoid (PerfectClosure K p)) with
    -- Porting note: added `zero_mul`, `mul_zero`
    zero_mul := fun a => by
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        a : PerfectClosure K p
        ⊢ Eq (HMul.hMul 0 a) 0
      -/
      refine Quot.inductionOn a fun ⟨m, x⟩ => ?_
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        a : PerfectClosure K p
        x✝ : Prod Nat K
        m : Nat
        x : K
        ⊢ Eq (HMul.hMul 0 (Quot.mk (PerfectClosure.R K p) { fst := m, snd := x })) 0
      -/
      rw [zero_def, quot_mk_eq_mk, mk_mul_mk]
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        a : PerfectClosure K p
        x✝ : Prod Nat K
        m : Nat
        x : K
        ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd { fst := 0, snd := 0 }.1 { fst  …
      -/
      simp only [zero_add, iterate_zero, id_eq, iterate_map_zero, zero_mul, mk_zero_right]
      /-
        🎉 no goals
      -/
    mul_zero := fun a => by
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        a : PerfectClosure K p
        ⊢ Eq (HMul.hMul a 0) 0
      -/
      refine Quot.inductionOn a fun ⟨m, x⟩ => ?_
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        a : PerfectClosure K p
        x✝ : Prod Nat K
        m : Nat
        x : K
        ⊢ Eq (HMul.hMul (Quot.mk (PerfectClosure.R K p) { fst := m, snd := x }) 0) 0
      -/
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (HMul.hMul (Quot.mk (PerfectClosure.R K p) { fst := m, snd := x }) (HAdd. …
            -/
      rw [zero_def, quot_mk_eq_mk, mk_mul_mk]
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd m (HAdd.hAdd n s), snd := HMul. …
            -/
      /-
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        a : PerfectClosure K p
        x✝ : Prod Nat K
        m : Nat
        x : K
        ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd { fst := m, snd := x }.1 { fst  …
      -/
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd m (HAdd.hAdd n s), snd := HMul. …
            -/
      simp only [zero_add, iterate_zero, id_eq, iterate_map_zero, mul_zero, mk_zero_right]
      /-
        🎉 no goals
      -/
    left_distrib := fun e f g =>
      Quot.inductionOn e fun ⟨m, x⟩ =>
        Quot.inductionOn f fun ⟨n, y⟩ =>
          Quot.inductionOn g fun ⟨s, z⟩ => by
            simp only [quot_mk_eq_mk, mk_add_mk, mk_mul_mk] -- Porting note: added this line
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (PerfectClosure.R K p) { fst := m, snd :=  …
            -/
            simp only [add_assoc, add_comm, add_left_comm]
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd (HAdd.hAdd m n) s, snd := HMul. …
            -/
            apply R.sound
            /-
              K : Type u
              inst✝² : CommRing K
              p : Nat
              inst✝¹ : Fact (Nat.Prime p)
              inst✝ : CharP K p
              e f g : PerfectClosure K p
              x✝² : Prod Nat K
              m : Nat
              x : K
              x✝¹ : Prod Nat K
              n : Nat
              y : K
              x✝ : Prod Nat K
              s : Nat
              z : K
              ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd s (HAdd.hAdd m n), snd := HMul. …
            -/
            simp only [iterate_map_mul, iterate_map_add, ← iterate_add_apply,
              mul_add, add_comm, add_left_comm]
    right_distrib := fun e f g =>
      Quot.inductionOn e fun ⟨m, x⟩ =>
        Quot.inductionOn f fun ⟨n, y⟩ =>
          Quot.inductionOn g fun ⟨s, z⟩ => by
            simp only [quot_mk_eq_mk, mk_add_mk, mk_mul_mk] -- Porting note: added this line
            simp only [add_assoc, add_comm _ s, add_left_comm _ s]
            apply R.sound
            simp only [iterate_map_mul, iterate_map_add, ← iterate_add_apply,
              add_mul, add_comm, add_left_comm] }


theorem mk_eq_iff (x y : ℕ × K) :
    mk K p x = mk K p y ↔ ∃ z, (frobenius K p)^[y.1 + z] x.2 = (frobenius K p)^[x.1 + z] y.2 := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x y : Prod Nat K
    ⊢ Iff (Eq (PerfectClosure.mk K p x) (PerfectClosure.mk K p y)) (Exists fun z = …
  -/
  constructor
    /-
      case mp
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x y : Prod Nat K
      ⊢ Eq (PerfectClosure.mk K p x) (PerfectClosure.mk K p y) → Exists fun z => Eq  …
    -/
  · intro H
    /-
      case mp
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x y : Prod Nat K
      H : Eq (PerfectClosure.mk K p x) (PerfectClosure.mk K p y)
      ⊢ Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd y.1 z) x.2) (N …
    -/
    replace H := Quot.eqvGen_exact H
    induction H with
    | rel x y H => cases' H with n x; exact ⟨0, rfl⟩
    | refl H => exact ⟨0, rfl⟩
    | symm x y H ih => cases' ih with w ih; exact ⟨w, ih.symm⟩
    | trans x y z H1 H2 ih1 ih2 =>
      cases' ih1 with z1 ih1
      cases' ih2 with z2 ih2
      exists z2 + (y.1 + z1)
      rw [← add_assoc, iterate_add_apply, ih1]
      rw [← iterate_add_apply, add_comm, iterate_add_apply, ih2]
      rw [← iterate_add_apply]
      simp only [add_comm, add_left_comm]
  /-
    case mpr
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x y : Prod Nat K
    ⊢ (Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd y.1 z) x.2) ( …
  -/
  intro H
  /-
    case mpr
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x y : Prod Nat K
    H : Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd y.1 z) x.2)  …
    ⊢ Eq (PerfectClosure.mk K p x) (PerfectClosure.mk K p y)
  -/
  cases' x with m x
  /-
    case mpr.mk
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    y : Prod Nat K
    m : Nat
    x : K
    H : Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd y.1 z) { fst …
    ⊢ Eq (PerfectClosure.mk K p { fst := m, snd := x }) (PerfectClosure.mk K p y)
  -/
  cases' y with n y
  /-
    case mpr.mk.mk
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    m : Nat
    x : K
    n : Nat
    y : K
    H : Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd { fst := n,  …
    ⊢ Eq (PerfectClosure.mk K p { fst := m, snd := x }) (PerfectClosure.mk K p { f …
  -/
  cases' H with z H; dsimp only at H
  /-
    case mpr.mk.mk.intro
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    m : Nat
    x : K
    n : Nat
    y : K
    z : Nat
    H : Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd n z) x) (Nat.iterate (⇑(frob …
    ⊢ Eq (PerfectClosure.mk K p { fst := m, snd := x }) (PerfectClosure.mk K p { f …
  -/
  rw [R.sound K p (n + z) m x _ rfl, R.sound K p (m + z) n y _ rfl, H]
  /-
    case mpr.mk.mk.intro
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    m : Nat
    x : K
    n : Nat
    y : K
    z : Nat
    H : Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd n z) x) (Nat.iterate (⇑(frob …
    ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd (HAdd.hAdd n z) m, snd := Nat.i …
  -/
  rw [add_assoc, add_comm, add_comm z]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_pow (x : ℕ × K) (n : ℕ) : mk K p x ^ n = mk K p (x.1, x.2 ^ n) := by
  induction n with
  | zero =>
    rw [pow_zero, pow_zero, one_def, mk_eq_iff]
    exact ⟨0, by simp_rw [← coe_iterateFrobenius, map_one]⟩
  | succ n ih =>
    rw [pow_succ, pow_succ, ih, mk_mul_mk, mk_eq_iff]
    exact ⟨0, by simp_rw [iterate_frobenius, add_zero, mul_pow, ← pow_mul,
      ← pow_add, mul_assoc, ← pow_add]⟩


theorem natCast (n x : ℕ) : (x : PerfectClosure K p) = mk K p (n, x) := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n x : Nat
    ⊢ Eq (↑x) (PerfectClosure.mk K p { fst := n, snd := ↑x })
  -/
  induction' n with n ih
    /-
      case zero
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : Nat
      ⊢ Eq (↑x) (PerfectClosure.mk K p { fst := 0, snd := ↑x })
    -/
  · induction' x with x ih
      /-
        case zero.zero
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        ⊢ Eq (↑0) (PerfectClosure.mk K p { fst := 0, snd := ↑0 })
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case zero.succ
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : Nat
      ih : Eq (↑x) (PerfectClosure.mk K p { fst := 0, snd := ↑x })
      ⊢ Eq (↑(HAdd.hAdd x 1)) (PerfectClosure.mk K p { fst := 0, snd := ↑(HAdd.hAdd  …
    -/
    rw [Nat.cast_succ, Nat.cast_succ, ih]
    /-
      case zero.succ
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : Nat
      ih : Eq (↑x) (PerfectClosure.mk K p { fst := 0, snd := ↑x })
      ⊢ Eq (HAdd.hAdd (PerfectClosure.mk K p { fst := 0, snd := ↑x }) 1) (PerfectClo …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x n : Nat
    ih : Eq (↑x) (PerfectClosure.mk K p { fst := n, snd := ↑x })
    ⊢ Eq (↑x) (PerfectClosure.mk K p { fst := HAdd.hAdd n 1, snd := ↑x })
  -/
  rw [ih]; apply Quot.sound
  -- Porting note: was `conv`
  suffices R K p (n, (x : K)) (Nat.succ n, frobenius K p (x : K)) by
    rwa [frobenius_natCast K p x] at this
  /-
    case succ.a
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x n : Nat
    ih : Eq (↑x) (PerfectClosure.mk K p { fst := n, snd := ↑x })
    ⊢ PerfectClosure.R K p { fst := n, snd := ↑x } { fst := n.succ, snd := (froben …
  -/
  apply R.intro
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast := natCast


theorem intCast (x : ℤ) : (x : PerfectClosure K p) = mk K p (0, x) := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x : Int
    ⊢ Eq (↑x) (PerfectClosure.mk K p { fst := 0, snd := ↑x })
  -/
                  /-
                    🎉 no goals
                  -/
  induction x <;> simp only [Int.ofNat_eq_coe, Int.cast_natCast, Int.cast_negSucc, natCast K p 0]
  /-
    case negSucc
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    a✝ : Nat
    ⊢ Eq (Neg.neg (PerfectClosure.mk K p { fst := 0, snd := ↑(HAdd.hAdd a✝ 1) }))  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias int_cast := intCast


theorem natCast_eq_iff (x y : ℕ) : (x : PerfectClosure K p) = y ↔ (x : K) = y := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x y : Nat
    ⊢ Iff (Eq ↑x ↑y) (Eq ↑x ↑y)
  -/
  constructor <;> intro H
    /-
      case mp
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x y : Nat
      H : Eq ↑x ↑y
      ⊢ Eq ↑x ↑y
    -/
  · rw [natCast K p 0, natCast K p 0, mk_eq_iff] at H
    /-
      case mp
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x y : Nat
      H : Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd { fst := 0,  …
      ⊢ Eq ↑x ↑y
    -/
    cases' H with z H
    /-
      case mp.intro
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x y z : Nat
      H : Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd { fst := 0, snd := ↑y }.1 z) …
      ⊢ Eq ↑x ↑y
    -/
    simpa only [zero_add, iterate_fixed (frobenius_natCast K p _)] using H
    /-
      🎉 no goals
    -/
  /-
    case mpr
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x y : Nat
    H : Eq ↑x ↑y
    ⊢ Eq ↑x ↑y
  -/
  rw [natCast K p 0, natCast K p 0, H]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_eq_iff := natCast_eq_iff


instance instCharP : CharP (PerfectClosure K p) p := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    ⊢ CharP (PerfectClosure K p) p
  -/
  constructor; intro x; rw [← CharP.cast_eq_zero_iff K]
  /-
    case cast_eq_zero_iff'
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x : Nat
    ⊢ Iff (Eq (↑x) 0) (Eq (↑x) 0)
  -/
  rw [← Nat.cast_zero, natCast_eq_iff, Nat.cast_zero]
  /-
    🎉 no goals
  -/


theorem frobenius_mk (x : ℕ × K) :
    (frobenius (PerfectClosure K p) p : PerfectClosure K p → PerfectClosure K p) (mk K p x) =
      mk _ _ (x.1, x.2 ^ p) := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x : Prod Nat K
    ⊢ Eq ((frobenius (PerfectClosure K p) p) (PerfectClosure.mk K p x)) (PerfectCl …
  -/
  simp only [frobenius_def]
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x : Prod Nat K
    ⊢ Eq (HPow.hPow (PerfectClosure.mk K p x) p) (PerfectClosure.mk K p { fst := x …
  -/
  exact mk_pow K p x p
  /-
    🎉 no goals
  -/


/-- Embedding of `K` into `PerfectClosure K p` -/
def of : K →+* PerfectClosure K p where
  toFun x := mk _ _ (0, x)
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl


theorem of_apply (x : K) : of K p x = mk _ _ (0, x) :=
  rfl


instance instReduced : IsReduced (PerfectClosure K p) where
  eq_zero x := induction_on x fun x ⟨n, h⟩ ↦ by
    replace h : mk K p x ^ p ^ n = 0 := by
      rw [← Nat.sub_add_cancel ((n.lt_pow_self (Fact.out : p.Prime).one_lt).le),
        pow_add, h, mul_zero]
    /-
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x✝¹ : PerfectClosure K p
      x : Prod Nat K
      x✝ : IsNilpotent (PerfectClosure.mk K p x)
      n : Nat
      h : Eq (HPow.hPow (PerfectClosure.mk K p x) (HPow.hPow p n)) 0
      ⊢ Eq (PerfectClosure.mk K p x) 0
    -/
    simp only [zero_def, mk_pow, mk_eq_iff, zero_add, ← coe_iterateFrobenius, map_zero] at h ⊢
    /-
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x✝¹ : PerfectClosure K p
      x : Prod Nat K
      x✝ : IsNilpotent (PerfectClosure.mk K p x)
      n : Nat
      h : Exists fun z => Eq ((iterateFrobenius K p z) (HPow.hPow x.2 (HPow.hPow p n …
      ⊢ Exists fun z => Eq ((iterateFrobenius K p z) x.2) 0
    -/
    obtain ⟨m, h⟩ := h
    /-
      case intro
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x✝¹ : PerfectClosure K p
      x : Prod Nat K
      x✝ : IsNilpotent (PerfectClosure.mk K p x)
      n m : Nat
      h : Eq ((iterateFrobenius K p m) (HPow.hPow x.2 (HPow.hPow p n))) 0
      ⊢ Exists fun z => Eq ((iterateFrobenius K p z) x.2) 0
    -/
    exact ⟨n + m, by simpa only [iterateFrobenius_def, pow_add, pow_mul] using h⟩
    /-
      🎉 no goals
    -/


instance instPerfectRing : PerfectRing (PerfectClosure K p) p where
  bijective_frobenius := by
    let f : PerfectClosure K p → PerfectClosure K p := fun e ↦
      liftOn e (fun x => mk K p (x.1 + 1, x.2)) fun x y H =>
      match x, y, H with
      | _, _, R.intro n x => Quot.sound (R.intro _ _)
    refine bijective_iff_has_inverse.mpr ⟨f, fun e ↦ induction_on e fun ⟨n, x⟩ ↦ ?_,
      fun e ↦ induction_on e fun ⟨n, x⟩ ↦ ?_⟩ <;>
      /-
        case refine_1
        K : Type u
        inst✝² : CommRing K
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : CharP K p
        f : PerfectClosure K p → PerfectClosure K p := fun e => e.liftOn (fun x => Per …
        e : PerfectClosure K p
        x✝ : Prod Nat K
        n : Nat
        x : K
        ⊢ Eq (f ((frobenius (PerfectClosure K p) p) (PerfectClosure.mk K p { fst := n, …
      -/
      /-
        🎉 no goals
      -/
      simp only [f, liftOn_mk, frobenius_mk, mk_succ_pow]
      /-
        🎉 no goals
      -/


@[simp]
theorem iterate_frobenius_mk (n : ℕ) (x : K) :
    (frobenius (PerfectClosure K p) p)^[n] (mk K p ⟨n, x⟩) = of K p x := by
  /-
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    x : K
    ⊢ Eq (Nat.iterate (⇑(frobenius (PerfectClosure K p) p)) n (PerfectClosure.mk K …
  -/
  induction' n with n ih
    /-
      case zero
      K : Type u
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : K
      ⊢ Eq (Nat.iterate (⇑(frobenius (PerfectClosure K p) p)) 0 (PerfectClosure.mk K …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    K : Type u
    inst✝² : CommRing K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    x : K
    n : Nat
    ih : Eq (Nat.iterate (⇑(frobenius (PerfectClosure K p) p)) n (PerfectClosure.m …
    ⊢ Eq (Nat.iterate (⇑(frobenius (PerfectClosure K p) p)) (HAdd.hAdd n 1) (Perfe …
  -/
  rw [iterate_succ_apply, ← ih, frobenius_mk, mk_succ_pow]
  /-
    🎉 no goals
  -/


/-- Given a ring `K` of characteristic `p` and a perfect ring `L` of the same characteristic,
any homomorphism `K →+* L` can be lifted to `PerfectClosure K p`. -/
noncomputable def lift (L : Type v) [CommSemiring L] [CharP L p] [PerfectRing L p] :
    (K →+* L) ≃ (PerfectClosure K p →+* L) where
  toFun f :=
    { toFun := by
        /-
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          ⊢ PerfectClosure K p → L
        -/
        refine fun e => liftOn e (fun x => (frobeniusEquiv L p).symm^[x.1] (f x.2)) ?_
        /-
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          e : PerfectClosure K p
          ⊢ ∀ (x y : Prod Nat K), PerfectClosure.R K p x y → Eq ((fun x => Nat.iterate ( …
        -/
        rintro - - ⟨n, x⟩
        /-
          case intro
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          e : PerfectClosure K p
          n : Nat
          x : K
          ⊢ Eq ((fun x => Nat.iterate (⇑(frobeniusEquiv L p).symm) x.1 (f x.2)) { fst := …
        -/
        simp [f.map_frobenius]
        /-
          🎉 no goals
        -/
      map_one' := f.map_one
      map_zero' := f.map_zero
      map_mul' := by
        /-
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          ⊢ ∀ (x y : PerfectClosure K p), Eq ({ toFun := fun e => e.liftOn (fun x => Nat …
        -/
        rintro ⟨n, x⟩ ⟨m, y⟩
        simp only [quot_mk_eq_mk, liftOn_mk, f.map_iterate_frobenius, mk_mul_mk, map_mul,
          iterate_map_mul]
        /-
          case mk.mk.mk.mk
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          x✝ : PerfectClosure K p
          n : Nat
          x : K
          y✝ : PerfectClosure K p
          m : Nat
          y : K
          ⊢ Eq (HMul.hMul (Nat.iterate (⇑(frobeniusEquiv L p).symm) (HAdd.hAdd n m) (Nat …
        -/
        have := LeftInverse.iterate (frobeniusEquiv_symm_apply_frobenius L p)
        /-
          case mk.mk.mk.mk
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          x✝ : PerfectClosure K p
          n : Nat
          x : K
          y✝ : PerfectClosure K p
          m : Nat
          y : K
          this : ∀ (n : Nat), Function.LeftInverse (Nat.iterate (⇑(frobeniusEquiv L p).s …
          ⊢ Eq (HMul.hMul (Nat.iterate (⇑(frobeniusEquiv L p).symm) (HAdd.hAdd n m) (Nat …
        -/
        rw [iterate_add_apply, this _ _, add_comm, iterate_add_apply, this _ _]
        /-
          🎉 no goals
        -/
      map_add' := by
        /-
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          ⊢ ∀ (x y : PerfectClosure K p), Eq ((↑{ toFun := fun e => e.liftOn (fun x => N …
        -/
        rintro ⟨n, x⟩ ⟨m, y⟩
        simp only [quot_mk_eq_mk, liftOn_mk, f.map_iterate_frobenius, mk_add_mk, map_add,
          iterate_map_add]
        /-
          case mk.mk.mk.mk
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          x✝ : PerfectClosure K p
          n : Nat
          x : K
          y✝ : PerfectClosure K p
          m : Nat
          y : K
          ⊢ Eq (HAdd.hAdd (Nat.iterate (⇑(frobeniusEquiv L p).symm) (HAdd.hAdd n m) (Nat …
        -/
        have := LeftInverse.iterate (frobeniusEquiv_symm_apply_frobenius L p)
        /-
          case mk.mk.mk.mk
          K : Type u
          inst✝⁵ : CommRing K
          p : Nat
          inst✝⁴ : Fact (Nat.Prime p)
          inst✝³ : CharP K p
          L : Type v
          inst✝² : CommSemiring L
          inst✝¹ : CharP L p
          inst✝ : PerfectRing L p
          f : RingHom K L
          x✝ : PerfectClosure K p
          n : Nat
          x : K
          y✝ : PerfectClosure K p
          m : Nat
          y : K
          this : ∀ (n : Nat), Function.LeftInverse (Nat.iterate (⇑(frobeniusEquiv L p).s …
          ⊢ Eq (HAdd.hAdd (Nat.iterate (⇑(frobeniusEquiv L p).symm) (HAdd.hAdd n m) (Nat …
        -/
        rw [iterate_add_apply, this _ _, add_comm n, iterate_add_apply, this _ _] }
        /-
          🎉 no goals
        -/
  invFun f := f.comp (of K p)
                   /-
                     K : Type u
                     inst✝⁵ : CommRing K
                     p : Nat
                     inst✝⁴ : Fact (Nat.Prime p)
                     inst✝³ : CharP K p
                     L : Type v
                     inst✝² : CommSemiring L
                     inst✝¹ : CharP L p
                     inst✝ : PerfectRing L p
                     f : RingHom K L
                     ⊢ Eq ((fun f => f.comp (PerfectClosure.of K p)) ((fun f => { toFun := fun e => …
                   -/
  left_inv f := by ext x; rfl
                          /-
                            🎉 no goals
                          -/
  right_inv f := by
    /-
      K : Type u
      inst✝⁵ : CommRing K
      p : Nat
      inst✝⁴ : Fact (Nat.Prime p)
      inst✝³ : CharP K p
      L : Type v
      inst✝² : CommSemiring L
      inst✝¹ : CharP L p
      inst✝ : PerfectRing L p
      f : RingHom (PerfectClosure K p) L
      ⊢ Eq ((fun f => { toFun := fun e => e.liftOn (fun x => Nat.iterate (⇑(frobeniu …
    -/
    ext ⟨n, x⟩
    simp only [quot_mk_eq_mk, RingHom.comp_apply, RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk,
      liftOn_mk]
    /-
      case a.mk.mk
      K : Type u
      inst✝⁵ : CommRing K
      p : Nat
      inst✝⁴ : Fact (Nat.Prime p)
      inst✝³ : CharP K p
      L : Type v
      inst✝² : CommSemiring L
      inst✝¹ : CharP L p
      inst✝ : PerfectRing L p
      f : RingHom (PerfectClosure K p) L
      x✝ : PerfectClosure K p
      n : Nat
      x : K
      ⊢ Eq (Nat.iterate (⇑(frobeniusEquiv L p).symm) n (f ((PerfectClosure.of K p) x …
    -/
    apply (injective_frobenius L p).iterate n
    rw [← f.map_iterate_frobenius, iterate_frobenius_mk,
      RightInverse.iterate (frobenius_apply_frobeniusEquiv_symm L p) n]


theorem eq_iff [CommRing K] [IsReduced K] (p : ℕ) [Fact p.Prime] [CharP K p] (x y : ℕ × K) :
    mk K p x = mk K p y ↔ (frobenius K p)^[y.1] x.2 = (frobenius K p)^[x.1] y.2 :=
  (mk_eq_iff K p x y).trans
                                                       /-
                                                         K : Type u
                                                         inst✝³ : CommRing K
                                                         inst✝² : IsReduced K
                                                         p : Nat
                                                         inst✝¹ : Fact (Nat.Prime p)
                                                         inst✝ : CharP K p
                                                         x y : Prod Nat K
                                                         x✝ : Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd y.1 z) x.2) …
                                                         z : Nat
                                                         H : Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd y.1 z) x.2) (Nat.iterate (⇑( …
                                                         ⊢ Eq (Nat.iterate (⇑(frobenius K p)) z (Nat.iterate (⇑(frobenius K p)) y.1 x.2 …
                                                       -/
    ⟨fun ⟨z, H⟩ => (frobenius_inj K p).iterate z <| by simpa only [add_comm, iterate_add] using H,
                                                       /-
                                                         🎉 no goals
                                                       -/
      fun H => ⟨0, H⟩⟩


instance instInv : Inv (PerfectClosure K p) :=
  ⟨Quot.lift (fun x : ℕ × K => Quot.mk (R K p) (x.1, x.2⁻¹)) fun x y (H : R K p x y) =>
      match x, y, H with
      | _, _, R.intro n x =>
        Quot.sound <| by
          /-
            K : Type u
            inst✝² : Field K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            x✝ y : Prod Nat K
            H : PerfectClosure.R K p x✝ y
            n : Nat
            x : K
            ⊢ PerfectClosure.R K p { fst := { fst := n, snd := x }.1, snd := Inv.inv { fst …
          -/
          simp only [frobenius_def]
          /-
            K : Type u
            inst✝² : Field K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            x✝ y : Prod Nat K
            H : PerfectClosure.R K p x✝ y
            n : Nat
            x : K
            ⊢ PerfectClosure.R K p { fst := n, snd := Inv.inv x } { fst := HAdd.hAdd n 1,  …
          -/
          rw [← inv_pow]
          /-
            K : Type u
            inst✝² : Field K
            p : Nat
            inst✝¹ : Fact (Nat.Prime p)
            inst✝ : CharP K p
            x✝ y : Prod Nat K
            H : PerfectClosure.R K p x✝ y
            n : Nat
            x : K
            ⊢ PerfectClosure.R K p { fst := n, snd := Inv.inv x } { fst := HAdd.hAdd n 1,  …
          -/
          apply R.intro⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem mk_inv (x : ℕ × K) : (mk K p x)⁻¹ = mk K p (x.1, x.2⁻¹) :=
  rfl

-- Porting note: added to avoid "unknown free variable" error

instance instDivisionRing : DivisionRing (PerfectClosure K p) where
  exists_pair_ne := ⟨0, 1, fun H => zero_ne_one ((eq_iff _ _ _ _).1 H)⟩
  mul_inv_cancel e := induction_on e fun ⟨m, x⟩ H ↦ by
    /-
      K : Type u
      inst✝² : Field K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      e : PerfectClosure K p
      x✝ : Prod Nat K
      m : Nat
      x : K
      H : Ne (PerfectClosure.mk K p { fst := m, snd := x }) 0
      ⊢ Eq (HMul.hMul (PerfectClosure.mk K p { fst := m, snd := x }) (Inv.inv (Perfe …
    -/
    have := mt (eq_iff _ _ _ _).2 H
    /-
      K : Type u
      inst✝² : Field K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      e : PerfectClosure K p
      x✝ : Prod Nat K
      m : Nat
      x : K
      H : Ne (PerfectClosure.mk K p { fst := m, snd := x }) 0
      this : Not (Eq (Nat.iterate ⇑(frobenius K p) { fst := 0, snd := 0 }.1 { fst := …
      ⊢ Eq (HMul.hMul (PerfectClosure.mk K p { fst := m, snd := x }) (Inv.inv (Perfe …
    -/
    rw [mk_inv, mk_mul_mk]
    /-
      K : Type u
      inst✝² : Field K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      e : PerfectClosure K p
      x✝ : Prod Nat K
      m : Nat
      x : K
      H : Ne (PerfectClosure.mk K p { fst := m, snd := x }) 0
      this : Not (Eq (Nat.iterate ⇑(frobenius K p) { fst := 0, snd := 0 }.1 { fst := …
      ⊢ Eq (PerfectClosure.mk K p { fst := HAdd.hAdd { fst := m, snd := x }.1 { fst  …
    -/
    refine (eq_iff K p _ _).2 ?_
    /-
      K : Type u
      inst✝² : Field K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      e : PerfectClosure K p
      x✝ : Prod Nat K
      m : Nat
      x : K
      H : Ne (PerfectClosure.mk K p { fst := m, snd := x }) 0
      this : Not (Eq (Nat.iterate ⇑(frobenius K p) { fst := 0, snd := 0 }.1 { fst := …
      ⊢ Eq (Nat.iterate ⇑(frobenius K p) { fst := 0, snd := 1 }.1 { fst := HAdd.hAdd …
    -/
    simp only [iterate_map_one, iterate_map_zero, iterate_zero_apply, ← iterate_map_mul] at this ⊢
    /-
      K : Type u
      inst✝² : Field K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      e : PerfectClosure K p
      x✝ : Prod Nat K
      m : Nat
      x : K
      H : Ne (PerfectClosure.mk K p { fst := m, snd := x }) 0
      this : Not (Eq x 0)
      ⊢ Eq (Nat.iterate (⇑(frobenius K p)) m (HMul.hMul x (Inv.inv x))) 1
    -/
    rw [mul_inv_cancel₀ this, iterate_map_one]
    /-
      🎉 no goals
    -/
                                              /-
                                                K : Type u
                                                inst✝² : Field K
                                                p : Nat
                                                inst✝¹ : Fact (Nat.Prime p)
                                                inst✝ : CharP K p
                                                ⊢ Eq { fst := { fst := 0, snd := 0 }.1, snd := Inv.inv { fst := 0, snd := 0 }. …
                                              -/
  inv_zero := congr_arg (Quot.mk (R K p)) (by rw [inv_zero])
                                              /-
                                                🎉 no goals
                                              -/
  nnqsmul := _
  nnqsmul_def := fun _ _  => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


instance instField : Field (PerfectClosure K p) :=
  { (inferInstance : DivisionRing (PerfectClosure K p)),
    (inferInstance : CommRing (PerfectClosure K p)) with }


instance instPerfectField : PerfectField (PerfectClosure K p) := PerfectRing.toPerfectField _ p


