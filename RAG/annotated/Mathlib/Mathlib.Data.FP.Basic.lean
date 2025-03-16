@[nolint docBlame]
def Int.shift2 (a b : ℕ) : ℤ → ℕ × ℕ
  | Int.ofNat e => (a <<< e, b)
  | Int.negSucc e => (a, b <<< e.succ)


@[nolint docBlame]
inductive RMode
  | NE -- round to nearest even
  deriving Inhabited


@[nolint docBlame]
class FloatCfg where
  (prec emax : ℕ)
  precPos : 0 < prec
  precMax : prec ≤ emax

@[nolint docBlame]
def prec :=
  C.prec


@[nolint docBlame]
def emax :=
  C.emax


@[nolint docBlame]
def emin : ℤ :=
  1 - C.emax


@[nolint docBlame]
def ValidFinite (e : ℤ) (m : ℕ) : Prop :=
  emin ≤ e + prec - 1 ∧ e + prec - 1 ≤ emax ∧ e = max (e + m.size - prec) emin


instance decValidFinite (e m) : Decidable (ValidFinite e m) := by
   /-
     C : FP.FloatCfg
     e : Int
     m : Nat
     ⊢ Decidable (FP.ValidFinite e m)
   -/
  (unfold ValidFinite; infer_instance)
                       /-
                         🎉 no goals
                       -/


@[nolint docBlame]
inductive Float
  | inf : Bool → Float
  | nan : Float
  | finite : Bool → ∀ e m, ValidFinite e m → Float


@[nolint docBlame]
def Float.isFinite : Float → Bool
  | Float.finite _ _ _ _ => true
  | _ => false


@[nolint docBlame]
def toRat : ∀ f : Float, f.isFinite → ℚ
  | Float.finite s e m _, _ =>
    let (n, d) := Int.shift2 m 1 e
    let r := mkRat n d
    if s then -r else r


theorem Float.Zero.valid : ValidFinite emin 0 :=
  ⟨by
    /-
      C : FP.FloatCfg
      ⊢ LE.le FP.emin (HSub.hSub (HAdd.hAdd FP.emin ↑FP.prec) 1)
    -/
    rw [add_sub_assoc]
    /-
      C : FP.FloatCfg
      ⊢ LE.le FP.emin (HAdd.hAdd FP.emin (HSub.hSub (↑FP.prec) 1))
    -/
    apply le_add_of_nonneg_right
    /-
      case h
      C : FP.FloatCfg
      ⊢ LE.le 0 (HSub.hSub (↑FP.prec) 1)
    -/
    apply sub_nonneg_of_le
    /-
      case h.a
      C : FP.FloatCfg
      ⊢ LE.le 1 ↑FP.prec
    -/
    apply Int.ofNat_le_ofNat_of_le
    /-
      case h.a.a
      C : FP.FloatCfg
      ⊢ LE.le 1 FP.prec
    -/
    exact C.precPos,
    /-
      🎉 no goals
    -/
    suffices prec ≤ 2 * emax by
      /-
        C : FP.FloatCfg
        this : LE.le FP.prec (HMul.hMul 2 FP.emax)
        ⊢ LE.le (HSub.hSub (HAdd.hAdd FP.emin ↑FP.prec) 1) ↑FP.emax
      -/
      rw [← Int.ofNat_le] at this
      /-
        C : FP.FloatCfg
        this : LE.le ↑FP.prec ↑(HMul.hMul 2 FP.emax)
        ⊢ LE.le (HSub.hSub (HAdd.hAdd FP.emin ↑FP.prec) 1) ↑FP.emax
      -/
      rw [← sub_nonneg] at *
      /-
        C : FP.FloatCfg
        this : LE.le 0 (HSub.hSub ↑(HMul.hMul 2 FP.emax) ↑FP.prec)
        ⊢ LE.le 0 (HSub.hSub (↑FP.emax) (HSub.hSub (HAdd.hAdd FP.emin ↑FP.prec) 1))
      -/
      simp only [emin, emax] at *
      /-
        C : FP.FloatCfg
        this : LE.le 0 (HSub.hSub ↑(HMul.hMul 2 FP.FloatCfg.emax) ↑FP.prec)
        ⊢ LE.le 0 (HSub.hSub (↑FP.FloatCfg.emax) (HSub.hSub (HAdd.hAdd (HSub.hSub 1 ↑F …
      -/
      omega
      /-
        🎉 no goals
      -/
    le_trans C.precMax (Nat.le_mul_of_pos_left _ Nat.zero_lt_two),
        /-
          C : FP.FloatCfg
          ⊢ Eq FP.emin (Max.max (HSub.hSub (HAdd.hAdd FP.emin ↑(Nat.size 0)) ↑FP.prec) F …
        -/
    by (rw [max_eq_right]; simp [sub_eq_add_neg, Int.ofNat_zero_le])⟩
                           /-
                             🎉 no goals
                           -/


@[nolint docBlame]
def Float.zero (s : Bool) : Float :=
  Float.finite s emin 0 Float.Zero.valid


instance : Inhabited Float :=
  ⟨Float.zero true⟩


@[nolint docBlame]
protected def Float.sign' : Float → Semiquot Bool
  | Float.inf s => pure s
  | Float.nan => ⊤
  | Float.finite s _ _ _ => pure s


@[nolint docBlame]
protected def Float.sign : Float → Bool
  | Float.inf s => s
  | Float.nan => false
  | Float.finite s _ _ _ => s


@[nolint docBlame]
protected def Float.isZero : Float → Bool
  | Float.finite _ _ 0 _ => true
  | _ => false


@[nolint docBlame]
protected def Float.neg : Float → Float
  | Float.inf s => Float.inf (not s)
  | Float.nan => Float.nan
  | Float.finite s e m f => Float.finite (not s) e m f


@[nolint docBlame]
def divNatLtTwoPow (n d : ℕ) : ℤ → Bool
  | Int.ofNat e => n < d <<< e
  | Int.negSucc e => n <<< e.succ < d


-- TODO(Mario): Prove these and drop 'unsafe'

@[nolint docBlame]
unsafe def ofPosRatDn (n : ℕ+) (d : ℕ+) : Float × Bool := by
  /-
    C : FP.FloatCfg
    n d : PNat
    ⊢ Prod FP.Float Bool
  -/
  let e₁ : ℤ := n.1.size - d.1.size - prec
  /-
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    ⊢ Prod FP.Float Bool
  -/
  cases' Int.shift2 d.1 n.1 (e₁ + prec) with d₁ n₁
  /-
    case mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    ⊢ Prod FP.Float Bool
  -/
  let e₂ := if n₁ < d₁ then e₁ - 1 else e₁
  /-
    case mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    e₂ : Int := ite (LT.lt n₁ d₁) (HSub.hSub e₁ 1) e₁
    ⊢ Prod FP.Float Bool
  -/
  let e₃ := max e₂ emin
  /-
    case mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    e₂ : Int := ite (LT.lt n₁ d₁) (HSub.hSub e₁ 1) e₁
    e₃ : Int := Max.max e₂ FP.emin
    ⊢ Prod FP.Float Bool
  -/
  cases' Int.shift2 d.1 n.1 (e₃ + prec) with d₂ n₂
  /-
    case mk.mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    e₂ : Int := ite (LT.lt n₁ d₁) (HSub.hSub e₁ 1) e₁
    e₃ : Int := Max.max e₂ FP.emin
    d₂ n₂ : Nat
    ⊢ Prod FP.Float Bool
  -/
  let r := mkRat n₂ d₂
  /-
    case mk.mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    e₂ : Int := ite (LT.lt n₁ d₁) (HSub.hSub e₁ 1) e₁
    e₃ : Int := Max.max e₂ FP.emin
    d₂ n₂ : Nat
    r : Rat := mkRat (↑n₂) d₂
    ⊢ Prod FP.Float Bool
  -/
  let m := r.floor
  /-
    case mk.mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    e₂ : Int := ite (LT.lt n₁ d₁) (HSub.hSub e₁ 1) e₁
    e₃ : Int := Max.max e₂ FP.emin
    d₂ n₂ : Nat
    r : Rat := mkRat (↑n₂) d₂
    m : Int := r.floor
    ⊢ Prod FP.Float Bool
  -/
  refine (Float.finite Bool.false e₃ (Int.toNat m) ?_, r.den = 1)
  /-
    case mk.mk
    C : FP.FloatCfg
    n d : PNat
    e₁ : Int := HSub.hSub (HSub.hSub ↑(↑n).size ↑(↑d).size) ↑FP.prec
    d₁ n₁ : Nat
    e₂ : Int := ite (LT.lt n₁ d₁) (HSub.hSub e₁ 1) e₁
    e₃ : Int := Max.max e₂ FP.emin
    d₂ n₂ : Nat
    r : Rat := mkRat (↑n₂) d₂
    m : Int := r.floor
    ⊢ FP.ValidFinite e₃ m.toNat
  -/
  exact lcProof
  /-
    🎉 no goals
  -/

-- Porting note: remove this line when you dropped 'lcProof'

set_option linter.unusedVariables false in
@[nolint docBlame]
unsafe def nextUpPos (e m) (v : ValidFinite e m) : Float :=
  let m' := m.succ
  if ss : m'.size = m.size then
                                /-
                                  C : FP.FloatCfg
                                  e : Int
                                  m : Nat
                                  v : FP.ValidFinite e m
                                  m' : Nat := m.succ
                                  ss : Eq m'.size m.size
                                  ⊢ FP.ValidFinite e m'
                                -/
    Float.finite false e m' (by unfold ValidFinite at *; rw [ss]; exact v)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  else if h : e = emax then Float.inf false else Float.finite false e.succ (Nat.div2 m') lcProof

-- Porting note: remove this line when you dropped 'lcProof'

set_option linter.unusedVariables false in
@[nolint docBlame]
unsafe def nextDnPos (e m) (v : ValidFinite e m) : Float :=
  match m with
  | 0 => nextUpPos _ _ Float.Zero.valid
  | Nat.succ m' =>
    -- Porting note: was `m'.size = m.size`
    if ss : m'.size = m'.succ.size then
                                  /-
                                    C : FP.FloatCfg
                                    e : Int
                                    m m' : Nat
                                    v : FP.ValidFinite e m'.succ
                                    ss : Eq m'.size m'.succ.size
                                    ⊢ FP.ValidFinite e m'
                                  -/
      Float.finite false e m' (by unfold ValidFinite at *; rw [ss]; exact v)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    else
      if h : e = emin then Float.finite false emin m' lcProof
      else Float.finite false e.pred (2 * m' + 1) lcProof


@[nolint docBlame]
unsafe def nextUp : Float → Float
  | Float.finite Bool.false e m f => nextUpPos e m f
  | Float.finite Bool.true e m f => Float.neg <| nextDnPos e m f
  | f => f


@[nolint docBlame]
unsafe def nextDn : Float → Float
  | Float.finite Bool.false e m f => nextDnPos e m f
  | Float.finite Bool.true e m f => Float.neg <| nextUpPos e m f
  | f => f


@[nolint docBlame]
unsafe def ofRatUp : ℚ → Float
  | ⟨0, _, _, _⟩ => Float.zero false
  | ⟨Nat.succ n, d, h, _⟩ =>
    let (f, exact) := ofPosRatDn n.succPNat ⟨d, Nat.pos_of_ne_zero h⟩
    if exact then f else nextUp f
  | ⟨Int.negSucc n, d, h, _⟩ => Float.neg (ofPosRatDn n.succPNat ⟨d, Nat.pos_of_ne_zero h⟩).1


@[nolint docBlame]
unsafe def ofRatDn (r : ℚ) : Float :=
  Float.neg <| ofRatUp (-r)


@[nolint docBlame]
unsafe def ofRat : RMode → ℚ → Float
  | RMode.NE, r =>
    let low := ofRatDn r
    let high := ofRatUp r
    if hf : high.isFinite then
      if r = toRat _ hf then high
      else
        if lf : low.isFinite then
          if r - toRat _ lf > toRat _ hf - r then high
          else
            if r - toRat _ lf < toRat _ hf - r then low
            else
              match low, lf with
              | Float.finite _ _ m _, _ => if 2 ∣ m then low else high
        else Float.inf true
    else Float.inf false


instance : Neg Float :=
  ⟨Float.neg⟩


@[nolint docBlame]
unsafe def add (mode : RMode) : Float → Float → Float
  | nan, _ => nan
  | _, nan => nan
  | inf Bool.true, inf Bool.false=> nan
  | inf Bool.false, inf Bool.true => nan
  | inf s₁, _ => inf s₁
  | _, inf s₂ => inf s₂
  | finite s₁ e₁ m₁ v₁, finite s₂ e₂ m₂ v₂ =>
    let f₁ := finite s₁ e₁ m₁ v₁
    let f₂ := finite s₂ e₂ m₂ v₂
    ofRat mode (toRat f₁ rfl + toRat f₂ rfl)


unsafe instance : Add Float :=
  ⟨Float.add RMode.NE⟩


@[nolint docBlame]
unsafe def sub (mode : RMode) (f1 f2 : Float) : Float :=
  add mode f1 (-f2)


unsafe instance : Sub Float :=
  ⟨Float.sub RMode.NE⟩


@[nolint docBlame]
unsafe def mul (mode : RMode) : Float → Float → Float
  | nan, _ => nan
  | _, nan => nan
  | inf s₁, f₂ => if f₂.isZero then nan else inf (xor s₁ f₂.sign)
  | f₁, inf s₂ => if f₁.isZero then nan else inf (xor f₁.sign s₂)
  | finite s₁ e₁ m₁ v₁, finite s₂ e₂ m₂ v₂ =>
    let f₁ := finite s₁ e₁ m₁ v₁
    let f₂ := finite s₂ e₂ m₂ v₂
    ofRat mode (toRat f₁ rfl * toRat f₂ rfl)


@[nolint docBlame]
unsafe def div (mode : RMode) : Float → Float → Float
  | nan, _ => nan
  | _, nan => nan
  | inf _, inf _ => nan
  | inf s₁, f₂ => inf (xor s₁ f₂.sign)
  | f₁, inf s₂ => zero (xor f₁.sign s₂)
  | finite s₁ e₁ m₁ v₁, finite s₂ e₂ m₂ v₂ =>
    let f₁ := finite s₁ e₁ m₁ v₁
    let f₂ := finite s₂ e₂ m₂ v₂
    if f₂.isZero then inf (xor s₁ s₂) else ofRat mode (toRat f₁ rfl / toRat f₂ rfl)


