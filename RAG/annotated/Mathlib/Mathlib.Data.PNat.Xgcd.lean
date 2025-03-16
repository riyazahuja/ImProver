/-- A term of `XgcdType` is a system of six naturals.  They should
 be thought of as representing the matrix
 [[w, x], [y, z]] = [[wp + 1, x], [y, zp + 1]]
 together with the vector [a, b] = [ap + 1, bp + 1].
-/
structure XgcdType where
  /-- `wp` is a variable which changes through the algorithm. -/
  wp : ℕ
  /-- `x` satisfies `a / d = w + x` at the final step. -/
  x : ℕ
  /-- `y` satisfies `b / d = z + y` at the final step. -/
  y : ℕ
  /-- `zp` is a variable which changes through the algorithm. -/
  zp : ℕ
  /-- `ap` is a variable which changes through the algorithm. -/
  ap : ℕ
  /-- `bp` is a variable which changes through the algorithm. -/
  bp : ℕ
  deriving Inhabited


instance : SizeOf XgcdType :=
  ⟨fun u => u.bp⟩


/-- The `Repr` instance converts terms to strings in a way that
 reflects the matrix/vector interpretation as above. -/
instance : Repr XgcdType where
  reprPrec
  | g, _ => s!"[[[{repr (g.wp + 1)}, {repr g.x}], \
                 [{repr g.y}, {repr (g.zp + 1)}]], \
                [{repr (g.ap + 1)}, {repr (g.bp + 1)}]]"


/-- Another `mk` using ℕ and ℕ+ -/
def mk' (w : ℕ+) (x : ℕ) (y : ℕ) (z : ℕ+) (a : ℕ+) (b : ℕ+) : XgcdType :=
  mk w.val.pred x y z.val.pred a.val.pred b.val.pred


/-- `w = wp + 1` -/
def w : ℕ+ :=
  succPNat u.wp


/-- `z = zp + 1` -/
def z : ℕ+ :=
  succPNat u.zp


/-- `a = ap + 1` -/
def a : ℕ+ :=
  succPNat u.ap


/-- `b = bp + 1` -/
def b : ℕ+ :=
  succPNat u.bp


/-- `r = a % b`: remainder -/
def r : ℕ :=
  (u.ap + 1) % (u.bp + 1)


/-- `q = ap / bp`: quotient -/
def q : ℕ :=
  (u.ap + 1) / (u.bp + 1)


/-- `qp = q - 1` -/
def qp : ℕ :=
  u.q - 1


/-- The map `v` gives the product of the matrix
 [[w, x], [y, z]] = [[wp + 1, x], [y, zp + 1]]
 and the vector [a, b] = [ap + 1, bp + 1].  The map
 `vp` gives [sp, tp] such that v = [sp + 1, tp + 1].
-/
def vp : ℕ × ℕ :=
  ⟨u.wp + u.x + u.ap + u.wp * u.ap + u.x * u.bp, u.y + u.zp + u.bp + u.y * u.ap + u.zp * u.bp⟩


/-- `v = [sp + 1, tp + 1]`, check `vp` -/
def v : ℕ × ℕ :=
  ⟨u.w * u.a + u.x * u.b, u.y * u.a + u.z * u.b⟩


/-- `succ₂ [t.1, t.2] = [t.1.succ, t.2.succ]` -/
def succ₂ (t : ℕ × ℕ) : ℕ × ℕ :=
  ⟨t.1.succ, t.2.succ⟩


theorem v_eq_succ_vp : u.v = succ₂ u.vp := by
  /-
    u : PNat.XgcdType
    ⊢ Eq u.v (PNat.XgcdType.succ₂ u.vp)
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  ext <;> dsimp [v, vp, w, z, a, b, succ₂] <;> ring_nf
                                               /-
                                                 🎉 no goals
                                               -/


/-- `IsSpecial` holds if the matrix has determinant one. -/
def IsSpecial : Prop :=
  u.wp + u.zp + u.wp * u.zp = u.x * u.y


/-- `IsSpecial'` is an alternative of `IsSpecial`. -/
def IsSpecial' : Prop :=
  u.w * u.z = succPNat (u.x * u.y)


theorem isSpecial_iff : u.IsSpecial ↔ u.IsSpecial' := by
  /-
    u : PNat.XgcdType
    ⊢ Iff u.IsSpecial u.IsSpecial'
  -/
  dsimp [IsSpecial, IsSpecial']
  /-
    u : PNat.XgcdType
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd u.wp u.zp) (HMul.hMul u.wp u.zp)) (HMul.hMul u …
  -/
  let ⟨wp, x, y, zp, ap, bp⟩ := u
  /-
    u : PNat.XgcdType
    wp x y zp ap bp : Nat
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd { wp := wp, x := x, y := y, zp := zp, ap := ap …
  -/
  constructor <;> intro h <;> simp only [w, succPNat, succ_eq_add_one, z] at * <;>
    /-
      case mp
      u : PNat.XgcdType
      wp x y zp ap bp : Nat
      h : Eq (HAdd.hAdd (HAdd.hAdd wp zp) (HMul.hMul wp zp)) (HMul.hMul x y)
      ⊢ Eq (HMul.hMul ⟨HAdd.hAdd wp 1, ⋯⟩ ⟨HAdd.hAdd zp 1, ⋯⟩) ⟨HAdd.hAdd (HMul.hMul …
    -/
    simp only [← coe_inj, mul_coe, mk_coe] at *
    /-
      case mp
      u : PNat.XgcdType
      wp x y zp ap bp : Nat
      h : Eq (HAdd.hAdd (HAdd.hAdd wp zp) (HMul.hMul wp zp)) (HMul.hMul x y)
      ⊢ Eq (HMul.hMul (HAdd.hAdd wp 1) (HAdd.hAdd zp 1)) (HAdd.hAdd (HMul.hMul x y) 1)
    -/
  · simp_all [← h]; ring
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      u : PNat.XgcdType
      wp x y zp ap bp : Nat
      h : Eq (HMul.hMul (HAdd.hAdd wp 1) (HAdd.hAdd zp 1)) (HAdd.hAdd (HMul.hMul x y …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd wp zp) (HMul.hMul wp zp)) (HMul.hMul x y)
    -/
  · simp [Nat.mul_add, Nat.add_mul, ← Nat.add_assoc] at h; rw [← h]; ring
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  -- Porting note: Old code has been removed as it was much more longer.


/-- `IsReduced` holds if the two entries in the vector are the
 same.  The reduction algorithm will produce a system with this
 property, whose product vector is the same as for the original
 system. -/
def IsReduced : Prop :=
  u.ap = u.bp


/-- `IsReduced'` is an alternative of `IsReduced`. -/
def IsReduced' : Prop :=
  u.a = u.b


theorem isReduced_iff : u.IsReduced ↔ u.IsReduced' :=
  succPNat_inj.symm


/-- `flip` flips the placement of variables during the algorithm. -/
def flip : XgcdType where
  wp := u.zp
  x := u.y
  y := u.x
  zp := u.wp
  ap := u.bp
  bp := u.ap


@[simp]
theorem flip_w : (flip u).w = u.z :=
  rfl


@[simp]
theorem flip_x : (flip u).x = u.y :=
  rfl


@[simp]
theorem flip_y : (flip u).y = u.x :=
  rfl


@[simp]
theorem flip_z : (flip u).z = u.w :=
  rfl


@[simp]
theorem flip_a : (flip u).a = u.b :=
  rfl


@[simp]
theorem flip_b : (flip u).b = u.a :=
  rfl


theorem flip_isReduced : (flip u).IsReduced ↔ u.IsReduced := by
  /-
    u : PNat.XgcdType
    ⊢ Iff u.flip.IsReduced u.IsReduced
  -/
  dsimp [IsReduced, flip]
  /-
    u : PNat.XgcdType
    ⊢ Iff (Eq u.bp u.ap) (Eq u.ap u.bp)
  -/
                              /-
                                🎉 no goals
                              -/
  constructor <;> intro h <;> exact h.symm
                              /-
                                🎉 no goals
                              -/


theorem flip_isSpecial : (flip u).IsSpecial ↔ u.IsSpecial := by
  /-
    u : PNat.XgcdType
    ⊢ Iff u.flip.IsSpecial u.IsSpecial
  -/
  dsimp [IsSpecial, flip]
  /-
    u : PNat.XgcdType
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd u.zp u.wp) (HMul.hMul u.zp u.wp)) (HMul.hMul u …
  -/
  rw [mul_comm u.x, mul_comm u.zp, add_comm u.zp]
  /-
    🎉 no goals
  -/


theorem flip_v : (flip u).v = u.v.swap := by
  /-
    u : PNat.XgcdType
    ⊢ Eq u.flip.v u.v.swap
  -/
  dsimp [v]
  /-
    u : PNat.XgcdType
    ⊢ Eq { fst := HAdd.hAdd (HMul.hMul ↑u.z ↑u.b) (HMul.hMul u.y ↑u.a), snd := HAd …
  -/
  ext
    /-
      case fst
      u : PNat.XgcdType
      ⊢ Eq { fst := HAdd.hAdd (HMul.hMul ↑u.z ↑u.b) (HMul.hMul u.y ↑u.a), snd := HAd …
    -/
  · simp only
    /-
      case fst
      u : PNat.XgcdType
      ⊢ Eq (HAdd.hAdd (HMul.hMul ↑u.z ↑u.b) (HMul.hMul u.y ↑u.a)) (HAdd.hAdd (HMul.h …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case snd
      u : PNat.XgcdType
      ⊢ Eq { fst := HAdd.hAdd (HMul.hMul ↑u.z ↑u.b) (HMul.hMul u.y ↑u.a), snd := HAd …
    -/
  · simp only
    /-
      case snd
      u : PNat.XgcdType
      ⊢ Eq (HAdd.hAdd (HMul.hMul u.x ↑u.b) (HMul.hMul ↑u.w ↑u.a)) (HAdd.hAdd (HMul.h …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Properties of division with remainder for a / b. -/
theorem rq_eq : u.r + (u.bp + 1) * u.q = u.ap + 1 :=
  Nat.mod_add_div (u.ap + 1) (u.bp + 1)


theorem qp_eq (hr : u.r = 0) : u.q = u.qp + 1 := by
  /-
    u : PNat.XgcdType
    hr : Eq u.r 0
    ⊢ Eq u.q (HAdd.hAdd u.qp 1)
  -/
  by_cases hq : u.q = 0
    /-
      case pos
      u : PNat.XgcdType
      hr : Eq u.r 0
      hq : Eq u.q 0
      ⊢ Eq u.q (HAdd.hAdd u.qp 1)
    -/
  · let h := u.rq_eq
    /-
      case pos
      u : PNat.XgcdType
      hr : Eq u.r 0
      hq : Eq u.q 0
      h : Eq (HAdd.hAdd u.r (HMul.hMul (HAdd.hAdd u.bp 1) u.q)) (HAdd.hAdd u.ap 1) : …
      ⊢ Eq u.q (HAdd.hAdd u.qp 1)
    -/
    rw [hr, hq, mul_zero, add_zero] at h
    /-
      case pos
      u : PNat.XgcdType
      hr : Eq u.r 0
      hq : Eq u.q 0
      h : Eq 0 (HAdd.hAdd u.ap 1)
      ⊢ Eq u.q (HAdd.hAdd u.qp 1)
    -/
    cases h
    /-
      🎉 no goals
    -/
    /-
      case neg
      u : PNat.XgcdType
      hr : Eq u.r 0
      hq : Not (Eq u.q 0)
      ⊢ Eq u.q (HAdd.hAdd u.qp 1)
    -/
  · exact (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero hq)).symm
    /-
      🎉 no goals
    -/


/-- The following function provides the starting point for
 our algorithm.  We will apply an iterative reduction process
 to it, which will produce a system satisfying IsReduced.
 The gcd can be read off from this final system.
-/
def start (a b : ℕ+) : XgcdType :=
  ⟨0, 0, 0, 0, a - 1, b - 1⟩


theorem start_isSpecial (a b : ℕ+) : (start a b).IsSpecial := by
  /-
    a b : PNat
    ⊢ (PNat.XgcdType.start a b).IsSpecial
  -/
  dsimp [start, IsSpecial]
  /-
    🎉 no goals
  -/


theorem start_v (a b : ℕ+) : (start a b).v = ⟨a, b⟩ := by
  /-
    a b : PNat
    ⊢ Eq (PNat.XgcdType.start a b).v { fst := ↑a, snd := ↑b }
  -/
  dsimp [start, v, XgcdType.a, XgcdType.b, w, z]
  /-
    a b : PNat
    ⊢ Eq { fst := HAdd.hAdd (HMul.hMul 1 (HAdd.hAdd (HSub.hSub (↑a) 1) 1)) (HMul.h …
  -/
  rw [one_mul, one_mul, zero_mul, zero_mul]
  /-
    a b : PNat
    ⊢ Eq { fst := HAdd.hAdd (HAdd.hAdd (HSub.hSub (↑a) 1) 1) 0, snd := HAdd.hAdd 0 …
  -/
  have := a.pos
  /-
    a b : PNat
    this : LT.lt 0 ↑a
    ⊢ Eq { fst := HAdd.hAdd (HAdd.hAdd (HSub.hSub (↑a) 1) 1) 0, snd := HAdd.hAdd 0 …
  -/
  have := b.pos
  /-
    a b : PNat
    this✝ : LT.lt 0 ↑a
    this : LT.lt 0 ↑b
    ⊢ Eq { fst := HAdd.hAdd (HAdd.hAdd (HSub.hSub (↑a) 1) 1) 0, snd := HAdd.hAdd 0 …
  -/
            /-
              🎉 no goals
            -/
  congr <;> omega
            /-
              🎉 no goals
            -/


/-- `finish` happens when the reducing process ends. -/
def finish : XgcdType :=
  XgcdType.mk u.wp ((u.wp + 1) * u.qp + u.x) u.y (u.y * u.qp + u.zp) u.bp u.bp


theorem finish_isReduced : u.finish.IsReduced := by
  /-
    u : PNat.XgcdType
    ⊢ u.finish.IsReduced
  -/
  dsimp [IsReduced]
  /-
    u : PNat.XgcdType
    ⊢ Eq u.finish.ap u.finish.bp
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem finish_isSpecial (hs : u.IsSpecial) : u.finish.IsSpecial := by
  /-
    u : PNat.XgcdType
    hs : u.IsSpecial
    ⊢ u.finish.IsSpecial
  -/
  dsimp [IsSpecial, finish] at hs ⊢
  /-
    u : PNat.XgcdType
    hs : Eq (HAdd.hAdd (HAdd.hAdd u.wp u.zp) (HMul.hMul u.wp u.zp)) (HMul.hMul u.x …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd u.wp (HAdd.hAdd (HMul.hMul u.y u.qp) u.zp)) (HMul.h …
  -/
  rw [add_mul _ _ u.y, add_comm _ (u.x * u.y), ← hs]
  /-
    u : PNat.XgcdType
    hs : Eq (HAdd.hAdd (HAdd.hAdd u.wp u.zp) (HMul.hMul u.wp u.zp)) (HMul.hMul u.x …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd u.wp (HAdd.hAdd (HMul.hMul u.y u.qp) u.zp)) (HMul.h …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem finish_v (hr : u.r = 0) : u.finish.v = u.v := by
  /-
    u : PNat.XgcdType
    hr : Eq u.r 0
    ⊢ Eq u.finish.v u.v
  -/
  let ha : u.r + u.b * u.q = u.a := u.rq_eq
  /-
    u : PNat.XgcdType
    hr : Eq u.r 0
    ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
    ⊢ Eq u.finish.v u.v
  -/
  rw [hr, zero_add] at ha
  /-
    u : PNat.XgcdType
    hr : Eq u.r 0
    ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
    ⊢ Eq u.finish.v u.v
  -/
  ext
    /-
      case fst
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      ⊢ Eq u.finish.v.1 u.v.1
    -/
  · change (u.wp + 1) * u.b + ((u.wp + 1) * u.qp + u.x) * u.b = u.w * u.a + u.x * u.b
    /-
      case fst
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd u.wp 1) ↑u.b) (HMul.hMul (HAdd.hAdd (HMu …
    -/
    have : u.wp + 1 = u.w := rfl
    /-
      case fst
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      this : Eq (HAdd.hAdd u.wp 1) ↑u.w
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd u.wp 1) ↑u.b) (HMul.hMul (HAdd.hAdd (HMu …
    -/
    rw [this, ← ha, u.qp_eq hr]
    /-
      case fst
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      this : Eq (HAdd.hAdd u.wp 1) ↑u.w
      ⊢ Eq (HAdd.hAdd (HMul.hMul ↑u.w ↑u.b) (HMul.hMul (HAdd.hAdd (HMul.hMul (↑u.w)  …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case snd
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      ⊢ Eq u.finish.v.2 u.v.2
    -/
  · change u.y * u.b + (u.y * u.qp + u.z) * u.b = u.y * u.a + u.z * u.b
    /-
      case snd
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      ⊢ Eq (HAdd.hAdd (HMul.hMul u.y ↑u.b) (HMul.hMul (HAdd.hAdd (HMul.hMul u.y u.qp …
    -/
    rw [← ha, u.qp_eq hr]
    /-
      case snd
      u : PNat.XgcdType
      hr : Eq u.r 0
      ha : Eq (HMul.hMul (↑u.b) u.q) ↑u.a
      ⊢ Eq (HAdd.hAdd (HMul.hMul u.y ↑u.b) (HMul.hMul (HAdd.hAdd (HMul.hMul u.y u.qp …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- This is the main reduction step, which is used when u.r ≠ 0, or
 equivalently b does not divide a. -/
def step : XgcdType :=
  XgcdType.mk (u.y * u.q + u.zp) u.y ((u.wp + 1) * u.q + u.x) u.wp u.bp (u.r - 1)


/-- We will apply the above step recursively.  The following result
 is used to ensure that the process terminates. -/
theorem step_wf (hr : u.r ≠ 0) : SizeOf.sizeOf u.step < SizeOf.sizeOf u := by
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    ⊢ LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
  -/
  change u.r - 1 < u.bp
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    ⊢ LT.lt (HSub.hSub u.r 1) u.bp
  -/
  have h₀ : u.r - 1 + 1 = u.r := Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero hr)
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    h₀ : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r
    ⊢ LT.lt (HSub.hSub u.r 1) u.bp
  -/
  have h₁ : u.r < u.bp + 1 := Nat.mod_lt (u.ap + 1) u.bp.succ_pos
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    h₀ : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r
    h₁ : LT.lt u.r (HAdd.hAdd u.bp 1)
    ⊢ LT.lt (HSub.hSub u.r 1) u.bp
  -/
  rw [← h₀] at h₁
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    h₀ : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r
    h₁ : LT.lt (HAdd.hAdd (HSub.hSub u.r 1) 1) (HAdd.hAdd u.bp 1)
    ⊢ LT.lt (HSub.hSub u.r 1) u.bp
  -/
  exact lt_of_succ_lt_succ h₁
  /-
    🎉 no goals
  -/


theorem step_isSpecial (hs : u.IsSpecial) : u.step.IsSpecial := by
  /-
    u : PNat.XgcdType
    hs : u.IsSpecial
    ⊢ u.step.IsSpecial
  -/
  dsimp [IsSpecial, step] at hs ⊢
  /-
    u : PNat.XgcdType
    hs : Eq (HAdd.hAdd (HAdd.hAdd u.wp u.zp) (HMul.hMul u.wp u.zp)) (HMul.hMul u.x …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul u.y u.q) u.zp) u.wp) (HMul.hM …
  -/
  rw [mul_add, mul_comm u.y u.x, ← hs]
  /-
    u : PNat.XgcdType
    hs : Eq (HAdd.hAdd (HAdd.hAdd u.wp u.zp) (HMul.hMul u.wp u.zp)) (HMul.hMul u.x …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul u.y u.q) u.zp) u.wp) (HMul.hM …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The reduction step does not change the product vector. -/
theorem step_v (hr : u.r ≠ 0) : u.step.v = u.v.swap := by
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    ⊢ Eq u.step.v u.v.swap
  -/
  let ha : u.r + u.b * u.q = u.a := u.rq_eq
  /-
    u : PNat.XgcdType
    hr : Ne u.r 0
    ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
    ⊢ Eq u.step.v u.v.swap
  -/
  let hr : u.r - 1 + 1 = u.r := (add_comm _ 1).trans (add_tsub_cancel_of_le (Nat.pos_of_ne_zero hr))
  /-
    u : PNat.XgcdType
    hr✝ : Ne u.r 0
    ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
    hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
    ⊢ Eq u.step.v u.v.swap
  -/
  ext
    /-
      case fst
      u : PNat.XgcdType
      hr✝ : Ne u.r 0
      ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
      hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
      ⊢ Eq u.step.v.1 u.v.swap.1
    -/
  · change ((u.y * u.q + u.z) * u.b + u.y * (u.r - 1 + 1) : ℕ) = u.y * u.a + u.z * u.b
    /-
      case fst
      u : PNat.XgcdType
      hr✝ : Ne u.r 0
      ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
      hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul u.y u.q) ↑u.z) ↑u.b) (HMul.hM …
    -/
    rw [← ha, hr]
    /-
      case fst
      u : PNat.XgcdType
      hr✝ : Ne u.r 0
      ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
      hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul u.y u.q) ↑u.z) ↑u.b) (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case snd
      u : PNat.XgcdType
      hr✝ : Ne u.r 0
      ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
      hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
      ⊢ Eq u.step.v.2 u.v.swap.2
    -/
  · change ((u.w * u.q + u.x) * u.b + u.w * (u.r - 1 + 1) : ℕ) = u.w * u.a + u.x * u.b
    /-
      case snd
      u : PNat.XgcdType
      hr✝ : Ne u.r 0
      ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
      hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul (↑u.w) u.q) u.x) ↑u.b) (HMul. …
    -/
    rw [← ha, hr]
    /-
      case snd
      u : PNat.XgcdType
      hr✝ : Ne u.r 0
      ha : Eq (HAdd.hAdd u.r (HMul.hMul (↑u.b) u.q)) ↑u.a := PNat.XgcdType.rq_eq u
      hr : Eq (HAdd.hAdd (HSub.hSub u.r 1) 1) u.r := Eq.trans (add_comm (HSub.hSub u …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul (↑u.w) u.q) u.x) ↑u.b) (HMul. …
    -/
    ring
    /-
      🎉 no goals
    -/

-- Porting note: removed 'have' and added decreasing_by to avoid lint errors

/-- We can now define the full reduction function, which applies
 step as long as possible, and then applies finish. Note that the
 "have" statement puts a fact in the local context, and the
 equation compiler uses this fact to help construct the full
 definition in terms of well-founded recursion.  The same fact
 needs to be introduced in all the inductive proofs of properties
 given below. -/
def reduce (u : XgcdType) : XgcdType :=
  dite (u.r = 0) (fun _ => u.finish) fun _h =>
    flip (reduce u.step)
/-
  u : PNat.XgcdType
  _h : Not (Eq u.r 0)
  ⊢ LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
-/
decreasing_by apply u.step_wf _h
/-
  🎉 no goals
-/


theorem reduce_a {u : XgcdType} (h : u.r = 0) : u.reduce = u.finish := by
  /-
    u : PNat.XgcdType
    h : Eq u.r 0
    ⊢ Eq u.reduce u.finish
  -/
  rw [reduce]
  /-
    u : PNat.XgcdType
    h : Eq u.r 0
    ⊢ Eq (dite (Eq u.r 0) (fun x => u.finish) fun _h => u.step.reduce.flip) u.finish
  -/
  exact if_pos h
  /-
    🎉 no goals
  -/


theorem reduce_b {u : XgcdType} (h : u.r ≠ 0) : u.reduce = u.step.reduce.flip := by
  /-
    u : PNat.XgcdType
    h : Ne u.r 0
    ⊢ Eq u.reduce u.step.reduce.flip
  -/
  rw [reduce]
  /-
    u : PNat.XgcdType
    h : Ne u.r 0
    ⊢ Eq (dite (Eq u.r 0) (fun x => u.finish) fun _h => u.step.reduce.flip) u.step …
  -/
  exact if_neg h
  /-
    🎉 no goals
  -/


theorem reduce_isReduced : ∀ u : XgcdType, u.reduce.IsReduced
  | u =>
    dite (u.r = 0)
      (fun h => by
        /-
          x✝ : PNat.XgcdType
          u : PNat.XgcdType := x✝
          h : Eq u.r 0
          ⊢ x✝.reduce.IsReduced
        -/
        rw [reduce_a h]
        /-
          x✝ : PNat.XgcdType
          u : PNat.XgcdType := x✝
          h : Eq u.r 0
          ⊢ u.finish.IsReduced
        -/
        exact u.finish_isReduced)
        /-
          🎉 no goals
        -/
      fun h => by
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        ⊢ x✝.reduce.IsReduced
      -/
      have : SizeOf.sizeOf u.step < SizeOf.sizeOf u := u.step_wf h
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        this : LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
        ⊢ x✝.reduce.IsReduced
      -/
      rw [reduce_b h, flip_isReduced]
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        this : LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
        ⊢ u.step.reduce.IsReduced
      -/
      apply reduce_isReduced
      /-
        🎉 no goals
      -/


theorem reduce_isReduced' (u : XgcdType) : u.reduce.IsReduced' :=
  (isReduced_iff _).mp u.reduce_isReduced


theorem reduce_isSpecial : ∀ u : XgcdType, u.IsSpecial → u.reduce.IsSpecial
  | u =>
    dite (u.r = 0)
      (fun h hs => by
        /-
          x✝ : PNat.XgcdType
          u : PNat.XgcdType := x✝
          h : Eq u.r 0
          hs : x✝.IsSpecial
          ⊢ x✝.reduce.IsSpecial
        -/
        rw [reduce_a h]
        /-
          x✝ : PNat.XgcdType
          u : PNat.XgcdType := x✝
          h : Eq u.r 0
          hs : x✝.IsSpecial
          ⊢ u.finish.IsSpecial
        -/
        exact u.finish_isSpecial hs)
        /-
          🎉 no goals
        -/
      fun h hs => by
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        hs : x✝.IsSpecial
        ⊢ x✝.reduce.IsSpecial
      -/
      have : SizeOf.sizeOf u.step < SizeOf.sizeOf u := u.step_wf h
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        hs : x✝.IsSpecial
        this : LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
        ⊢ x✝.reduce.IsSpecial
      -/
      rw [reduce_b h]
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        hs : x✝.IsSpecial
        this : LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
        ⊢ u.step.reduce.flip.IsSpecial
      -/
      exact (flip_isSpecial _).mpr (reduce_isSpecial _ (u.step_isSpecial hs))
      /-
        🎉 no goals
      -/


theorem reduce_isSpecial' (u : XgcdType) (hs : u.IsSpecial) : u.reduce.IsSpecial' :=
  (isSpecial_iff _).mp (u.reduce_isSpecial hs)


theorem reduce_v : ∀ u : XgcdType, u.reduce.v = u.v
  | u =>
                                /-
                                  x✝ : PNat.XgcdType
                                  u : PNat.XgcdType := x✝
                                  h : Eq u.r 0
                                  ⊢ Eq x✝.reduce.v x✝.v
                                -/
    dite (u.r = 0) (fun h => by rw [reduce_a h, finish_v u h]) fun h => by
                                /-
                                  🎉 no goals
                                -/
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        ⊢ Eq x✝.reduce.v x✝.v
      -/
      have : SizeOf.sizeOf u.step < SizeOf.sizeOf u := u.step_wf h
      /-
        x✝ : PNat.XgcdType
        u : PNat.XgcdType := x✝
        h : Not (Eq u.r 0)
        this : LT.lt (SizeOf.sizeOf u.step) (SizeOf.sizeOf u)
        ⊢ Eq x✝.reduce.v x✝.v
      -/
      rw [reduce_b h, flip_v, reduce_v (step u), step_v u h, Prod.swap_swap]
      /-
        🎉 no goals
      -/


/-- Extended Euclidean algorithm -/
def xgcd : XgcdType :=
  (XgcdType.start a b).reduce


/-- `gcdD a b = gcd a b` -/
def gcdD : ℕ+ :=
  (xgcd a b).a


/-- Final value of `w` -/
def gcdW : ℕ+ :=
  (xgcd a b).w


/-- Final value of `x` -/
def gcdX : ℕ :=
  (xgcd a b).x


/-- Final value of `y` -/
def gcdY : ℕ :=
  (xgcd a b).y


/-- Final value of `z` -/
def gcdZ : ℕ+ :=
  (xgcd a b).z


/-- Final value of `a / d` -/
def gcdA' : ℕ+ :=
  succPNat ((xgcd a b).wp + (xgcd a b).x)


/-- Final value of `b / d` -/
def gcdB' : ℕ+ :=
  succPNat ((xgcd a b).y + (xgcd a b).zp)


theorem gcdA'_coe : (gcdA' a b : ℕ) = gcdW a b + gcdX a b := by
  /-
    a b : PNat
    ⊢ Eq (↑(a.gcdA' b)) (HAdd.hAdd (↑(a.gcdW b)) (a.gcdX b))
  -/
  dsimp [gcdA', gcdX, gcdW, XgcdType.w]
  /-
    a b : PNat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (a.xgcd b).wp (a.xgcd b).x) 1) (HAdd.hAdd (HAdd.hAd …
  -/
  rw [add_right_comm]
  /-
    🎉 no goals
  -/


theorem gcdB'_coe : (gcdB' a b : ℕ) = gcdY a b + gcdZ a b := by
  /-
    a b : PNat
    ⊢ Eq (↑(a.gcdB' b)) (HAdd.hAdd (a.gcdY b) ↑(a.gcdZ b))
  -/
  dsimp [gcdB', gcdY, gcdZ, XgcdType.z]
  /-
    a b : PNat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (a.xgcd b).y (a.xgcd b).zp) 1) (HAdd.hAdd (a.xgcd b …
  -/
  rw [add_assoc]
  /-
    🎉 no goals
  -/


theorem gcd_props :
    let d := gcdD a b
    let w := gcdW a b
    let x := gcdX a b
    let y := gcdY a b
    let z := gcdZ a b
    let a' := gcdA' a b
    let b' := gcdB' a b
    w * z = succPNat (x * y) ∧
      a = a' * d ∧
        b = b' * d ∧
          z * a' = succPNat (x * b') ∧
            w * b' = succPNat (y * a') ∧ (z * a : ℕ) = x * b + d ∧ (w * b : ℕ) = y * a + d := by
  /-
    a b : PNat
    ⊢ let d := a.gcdD b;
      let w := a.gcdW b;
      let x := a.gcdX b;
      let y := a.gcdY b;
      let z := a.gcdZ b;
      let a' := a.gcdA' b;
      let b' := a.gcdB' b;
      And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  intros d w x y z a' b'
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  let u := XgcdType.start a b
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  let ur := u.reduce

  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  have _ : d = ur.a := rfl
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  have hb : d = ur.b := u.reduce_isReduced'
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    hb : Eq d ur.b
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  have ha' : (a' : ℕ) = w + x := gcdA'_coe a b
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  have hb' : (b' : ℕ) = y + z := gcdB'_coe a b
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  have hdet : w * z = succPNat (x * y) := u.reduce_isSpecial' rfl
  /-
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    ⊢ And (Eq (HMul.hMul w z) (HMul.hMul x y).succPNat) (And (Eq a (HMul.hMul a' d …
  -/
  constructor
    /-
      case left
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      ⊢ Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    -/
  · exact hdet
    /-
      🎉 no goals
    -/
  /-
    case right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    ⊢ And (Eq a (HMul.hMul a' d)) (And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul …
  -/
  have hdet' : (w * z : ℕ) = x * y + 1 := by rw [← mul_coe, hdet, succPNat_coe]
  /-
    case right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    ⊢ And (Eq a (HMul.hMul a' d)) (And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul …
  -/
  have _ : u.v = ⟨a, b⟩ := XgcdType.start_v a b
  let hv : Prod.mk (w * d + x * ur.b : ℕ) (y * d + z * ur.b : ℕ) = ⟨a, b⟩ :=
    u.reduce_v.trans (XgcdType.start_v a b)
  /-
    case right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HAdd.hAdd (HMul.hMul ↑w ↑d) (HMul.hMul x ↑ur.b), snd := HAdd. …
    ⊢ And (Eq a (HMul.hMul a' d)) (And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul …
  -/
  rw [← hb, ← add_mul, ← add_mul, ← ha', ← hb'] at hv
  /-
    case right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ⊢ And (Eq a (HMul.hMul a' d)) (And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul …
  -/
  have ha'' : (a : ℕ) = a' * d := (congr_arg Prod.fst hv).symm
  /-
    case right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    ⊢ And (Eq a (HMul.hMul a' d)) (And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul …
  -/
  have hb'' : (b : ℕ) = b' * d := (congr_arg Prod.snd hv).symm
  /-
    case right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    ⊢ And (Eq a (HMul.hMul a' d)) (And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul …
  -/
  constructor
    /-
      case right.left
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝¹ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
      x✝ : Eq u.v { fst := ↑a, snd := ↑b }
      hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
      ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
      hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
      ⊢ Eq a (HMul.hMul a' d)
    -/
  · exact eq ha''
    /-
      🎉 no goals
    -/
  /-
    case right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    ⊢ And (Eq b (HMul.hMul b' d)) (And (Eq (HMul.hMul z a') (HMul.hMul x ↑b').succ …
  -/
  constructor
    /-
      case right.right.left
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝¹ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
      x✝ : Eq u.v { fst := ↑a, snd := ↑b }
      hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
      ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
      hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
      ⊢ Eq b (HMul.hMul b' d)
    -/
  · exact eq hb''
    /-
      🎉 no goals
    -/
  have hza' : (z * a' : ℕ) = x * b' + 1 := by
    rw [ha', hb', mul_add, mul_add, mul_comm (z : ℕ), hdet']
    ring
  have hwb' : (w * b' : ℕ) = y * a' + 1 := by
    rw [ha', hb', mul_add, mul_add, hdet']
    ring
  /-
    case right.right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
    hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
    ⊢ And (Eq (HMul.hMul z a') (HMul.hMul x ↑b').succPNat) (And (Eq (HMul.hMul w b …
  -/
  constructor
    /-
      case right.right.right.left
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝¹ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
      x✝ : Eq u.v { fst := ↑a, snd := ↑b }
      hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
      ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
      hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
      hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
      hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
      ⊢ Eq (HMul.hMul z a') (HMul.hMul x ↑b').succPNat
    -/
  · apply eq
    /-
      case right.right.right.left.a
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝¹ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
      x✝ : Eq u.v { fst := ↑a, snd := ↑b }
      hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
      ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
      hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
      hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
      hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
      ⊢ Eq ↑(HMul.hMul z a') ↑(HMul.hMul x ↑b').succPNat
    -/
    rw [succPNat_coe, Nat.succ_eq_add_one, mul_coe, hza']
    /-
      🎉 no goals
    -/
  /-
    case right.right.right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
    hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
    ⊢ And (Eq (HMul.hMul w b') (HMul.hMul y ↑a').succPNat) (And (Eq (HMul.hMul ↑z  …
  -/
  constructor
    /-
      case right.right.right.right.left
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝¹ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
      x✝ : Eq u.v { fst := ↑a, snd := ↑b }
      hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
      ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
      hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
      hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
      hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
      ⊢ Eq (HMul.hMul w b') (HMul.hMul y ↑a').succPNat
    -/
  · apply eq
    /-
      case right.right.right.right.left.a
      a b : PNat
      d : PNat := a.gcdD b
      w : PNat := a.gcdW b
      x : Nat := a.gcdX b
      y : Nat := a.gcdY b
      z : PNat := a.gcdZ b
      a' : PNat := a.gcdA' b
      b' : PNat := a.gcdB' b
      u : PNat.XgcdType := PNat.XgcdType.start a b
      ur : PNat.XgcdType := u.reduce
      x✝¹ : Eq d ur.a
      hb : Eq d ur.b
      ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
      hb' : Eq (↑b') (HAdd.hAdd y ↑z)
      hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
      hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
      x✝ : Eq u.v { fst := ↑a, snd := ↑b }
      hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
      ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
      hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
      hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
      hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
      ⊢ Eq ↑(HMul.hMul w b') ↑(HMul.hMul y ↑a').succPNat
    -/
    rw [succPNat_coe, Nat.succ_eq_add_one, mul_coe, hwb']
    /-
      🎉 no goals
    -/
  /-
    case right.right.right.right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
    hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
    ⊢ And (Eq (HMul.hMul ↑z ↑a) (HAdd.hAdd (HMul.hMul x ↑b) ↑d)) (Eq (HMul.hMul ↑w …
  -/
  rw [ha'', hb'']
  /-
    case right.right.right.right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
    hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
    ⊢ And (Eq (HMul.hMul (↑z) (HMul.hMul ↑a' ↑d)) (HAdd.hAdd (HMul.hMul x (HMul.hM …
  -/
  repeat rw [← @mul_assoc]
  /-
    case right.right.right.right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
    hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
    ⊢ And (Eq (HMul.hMul (HMul.hMul ↑z ↑a') ↑d) (HAdd.hAdd (HMul.hMul (HMul.hMul x …
  -/
  rw [hza', hwb']
  /-
    case right.right.right.right.right
    a b : PNat
    d : PNat := a.gcdD b
    w : PNat := a.gcdW b
    x : Nat := a.gcdX b
    y : Nat := a.gcdY b
    z : PNat := a.gcdZ b
    a' : PNat := a.gcdA' b
    b' : PNat := a.gcdB' b
    u : PNat.XgcdType := PNat.XgcdType.start a b
    ur : PNat.XgcdType := u.reduce
    x✝¹ : Eq d ur.a
    hb : Eq d ur.b
    ha' : Eq (↑a') (HAdd.hAdd (↑w) x)
    hb' : Eq (↑b') (HAdd.hAdd y ↑z)
    hdet : Eq (HMul.hMul w z) (HMul.hMul x y).succPNat
    hdet' : Eq (HMul.hMul ↑w ↑z) (HAdd.hAdd (HMul.hMul x y) 1)
    x✝ : Eq u.v { fst := ↑a, snd := ↑b }
    hv : Eq { fst := HMul.hMul ↑a' ↑d, snd := HMul.hMul ↑b' ↑d } { fst := ↑a, snd  …
    ha'' : Eq (↑a) (HMul.hMul ↑a' ↑d)
    hb'' : Eq (↑b) (HMul.hMul ↑b' ↑d)
    hza' : Eq (HMul.hMul ↑z ↑a') (HAdd.hAdd (HMul.hMul x ↑b') 1)
    hwb' : Eq (HMul.hMul ↑w ↑b') (HAdd.hAdd (HMul.hMul y ↑a') 1)
    ⊢ And (Eq (HMul.hMul (HAdd.hAdd (HMul.hMul x ↑b') 1) ↑d) (HAdd.hAdd (HMul.hMul …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> ring
                  /-
                    🎉 no goals
                  -/


theorem gcd_eq : gcdD a b = gcd a b := by
  /-
    a b : PNat
    ⊢ Eq (a.gcdD b) (a.gcd b)
  -/
  rcases gcd_props a b with ⟨_, h₁, h₂, _, _, h₅, _⟩
  /-
    case intro.intro.intro.intro.intro.intro
    a b : PNat
    left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
    h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
    h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
    left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
    left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
    h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
    right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
    ⊢ Eq (a.gcdD b) (a.gcd b)
  -/
  apply dvd_antisymm
    /-
      case intro.intro.intro.intro.intro.intro.a
      a b : PNat
      left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
      h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
      h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
      left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
      left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
      h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
      right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
      ⊢ Dvd.dvd (a.gcdD b) (a.gcd b)
    -/
  · apply dvd_gcd
      /-
        case intro.intro.intro.intro.intro.intro.a.hm
        a b : PNat
        left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
        h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
        h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
        left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
        left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
        h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
        right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
        ⊢ Dvd.dvd (a.gcdD b) a
      -/
    · exact Dvd.intro (gcdA' a b) (h₁.trans (mul_comm _ _)).symm
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.a.hn
        a b : PNat
        left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
        h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
        h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
        left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
        left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
        h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
        right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
        ⊢ Dvd.dvd (a.gcdD b) b
      -/
    · exact Dvd.intro (gcdB' a b) (h₂.trans (mul_comm _ _)).symm
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.intro.a
      a b : PNat
      left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
      h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
      h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
      left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
      left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
      h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
      right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
      ⊢ Dvd.dvd (a.gcd b) (a.gcdD b)
    -/
  · have h₇ : (gcd a b : ℕ) ∣ gcdZ a b * a := (Nat.gcd_dvd_left a b).trans (dvd_mul_left _ _)
    /-
      case intro.intro.intro.intro.intro.intro.a
      a b : PNat
      left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
      h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
      h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
      left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
      left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
      h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
      right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
      h₇ : Dvd.dvd (↑(a.gcd b)) (HMul.hMul ↑(a.gcdZ b) ↑a)
      ⊢ Dvd.dvd (a.gcd b) (a.gcdD b)
    -/
    have h₈ : (gcd a b : ℕ) ∣ gcdX a b * b := (Nat.gcd_dvd_right a b).trans (dvd_mul_left _ _)
    /-
      case intro.intro.intro.intro.intro.intro.a
      a b : PNat
      left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
      h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
      h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
      left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
      left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
      h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
      right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
      h₇ : Dvd.dvd (↑(a.gcd b)) (HMul.hMul ↑(a.gcdZ b) ↑a)
      h₈ : Dvd.dvd (↑(a.gcd b)) (HMul.hMul (a.gcdX b) ↑b)
      ⊢ Dvd.dvd (a.gcd b) (a.gcdD b)
    -/
    rw [h₅] at h₇
    /-
      case intro.intro.intro.intro.intro.intro.a
      a b : PNat
      left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
      h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
      h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
      left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
      left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
      h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
      right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
      h₇ : Dvd.dvd (↑(a.gcd b)) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gcdD b))
      h₈ : Dvd.dvd (↑(a.gcd b)) (HMul.hMul (a.gcdX b) ↑b)
      ⊢ Dvd.dvd (a.gcd b) (a.gcdD b)
    -/
    rw [dvd_iff]
    /-
      case intro.intro.intro.intro.intro.intro.a
      a b : PNat
      left✝² : Eq (HMul.hMul (a.gcdW b) (a.gcdZ b)) (HMul.hMul (a.gcdX b) (a.gcdY b) …
      h₁ : Eq a (HMul.hMul (a.gcdA' b) (a.gcdD b))
      h₂ : Eq b (HMul.hMul (a.gcdB' b) (a.gcdD b))
      left✝¹ : Eq (HMul.hMul (a.gcdZ b) (a.gcdA' b)) (HMul.hMul (a.gcdX b) ↑(a.gcdB' …
      left✝ : Eq (HMul.hMul (a.gcdW b) (a.gcdB' b)) (HMul.hMul (a.gcdY b) ↑(a.gcdA'  …
      h₅ : Eq (HMul.hMul ↑(a.gcdZ b) ↑a) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gc …
      right✝ : Eq (HMul.hMul ↑(a.gcdW b) ↑b) (HAdd.hAdd (HMul.hMul (a.gcdY b) ↑a) ↑( …
      h₇ : Dvd.dvd (↑(a.gcd b)) (HAdd.hAdd (HMul.hMul (a.gcdX b) ↑b) ↑(a.gcdD b))
      h₈ : Dvd.dvd (↑(a.gcd b)) (HMul.hMul (a.gcdX b) ↑b)
      ⊢ Dvd.dvd ↑(a.gcd b) ↑(a.gcdD b)
    -/
    exact (Nat.dvd_add_iff_right h₈).mpr h₇
    /-
      🎉 no goals
    -/


theorem gcd_det_eq : gcdW a b * gcdZ a b = succPNat (gcdX a b * gcdY a b) :=
  (gcd_props a b).1


theorem gcd_a_eq : a = gcdA' a b * gcd a b :=
  gcd_eq a b ▸ (gcd_props a b).2.1


theorem gcd_b_eq : b = gcdB' a b * gcd a b :=
  gcd_eq a b ▸ (gcd_props a b).2.2.1


theorem gcd_rel_left' : gcdZ a b * gcdA' a b = succPNat (gcdX a b * gcdB' a b) :=
  (gcd_props a b).2.2.2.1


theorem gcd_rel_right' : gcdW a b * gcdB' a b = succPNat (gcdY a b * gcdA' a b) :=
  (gcd_props a b).2.2.2.2.1


theorem gcd_rel_left : (gcdZ a b * a : ℕ) = gcdX a b * b + gcd a b :=
  gcd_eq a b ▸ (gcd_props a b).2.2.2.2.2.1


theorem gcd_rel_right : (gcdW a b * b : ℕ) = gcdY a b * a + gcd a b :=
  gcd_eq a b ▸ (gcd_props a b).2.2.2.2.2.2


