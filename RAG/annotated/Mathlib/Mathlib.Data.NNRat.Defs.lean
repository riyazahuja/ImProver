instance Rat.instZeroLEOneClass : ZeroLEOneClass ℚ where
  zero_le_one := rfl


instance Rat.instPosMulMono : PosMulMono ℚ where
  elim := fun r p q h => by
    /-
      r : Subtype fun x => LE.le 0 x
      p q : Rat
      h : LE.le p q
      ⊢ LE.le (HMul.hMul (↑r) p) (HMul.hMul (↑r) q)
    -/
    simp only [mul_comm]
    /-
      r : Subtype fun x => LE.le 0 x
      p q : Rat
      h : LE.le p q
      ⊢ LE.le (HMul.hMul p ↑r) (HMul.hMul q ↑r)
    -/
    simpa [sub_mul, sub_nonneg] using Rat.mul_nonneg (sub_nonneg.2 h) r.2
    /-
      🎉 no goals
    -/


deriving instance CommSemiring for NNRat

deriving instance LinearOrder for NNRat

deriving instance Sub for NNRat

deriving instance Inhabited for NNRat


                                                                           /-
                                                                             p q : NNRat
                                                                             ⊢ Ne 1 0
                                                                           -/
instance instNontrivial : Nontrivial ℚ≥0 where exists_pair_ne := ⟨1, 0, by decide⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

instance instOrderBot : OrderBot ℚ≥0 where
  bot := 0
  bot_le q := q.2


@[simp] lemma val_eq_cast (q : ℚ≥0) : q.1 = q := rfl


instance instCharZero : CharZero ℚ≥0 where
                               /-
                                 p q : NNRat
                                 a b : Nat
                                 hab : Eq ↑a ↑b
                                 ⊢ Eq a b
                               -/
  cast_injective a b hab := by simpa using congr_arg num hab
                               /-
                                 🎉 no goals
                               -/


instance canLift : CanLift ℚ ℚ≥0 (↑) fun q ↦ 0 ≤ q where
  prf q hq := ⟨⟨q, hq⟩, rfl⟩


@[ext]
theorem ext : (p : ℚ) = (q : ℚ) → p = q :=
  Subtype.ext


protected theorem coe_injective : Injective ((↑) : ℚ≥0 → ℚ) :=
  Subtype.coe_injective

-- See note [specialised high priority simp lemma]

@[simp high, norm_cast]
theorem coe_inj : (p : ℚ) = q ↔ p = q :=
  Subtype.coe_inj


theorem ne_iff {x y : ℚ≥0} : (x : ℚ) ≠ (y : ℚ) ↔ x ≠ y :=
  NNRat.coe_inj.not

-- TODO: We have to write `NNRat.cast` explicitly, else the statement picks up `Subtype.val` instead

@[simp, norm_cast] lemma coe_mk (q : ℚ) (hq) : NNRat.cast ⟨q, hq⟩ = q := rfl


lemma «forall» {p : ℚ≥0 → Prop} : (∀ q, p q) ↔ ∀ q hq, p ⟨q, hq⟩ := Subtype.forall

lemma «exists» {p : ℚ≥0 → Prop} : (∃ q, p q) ↔ ∃ q hq, p ⟨q, hq⟩ := Subtype.exists


/-- Reinterpret a rational number `q` as a non-negative rational number. Returns `0` if `q ≤ 0`. -/
def _root_.Rat.toNNRat (q : ℚ) : ℚ≥0 :=
  ⟨max q 0, le_max_right _ _⟩


theorem _root_.Rat.coe_toNNRat (q : ℚ) (hq : 0 ≤ q) : (q.toNNRat : ℚ) = q :=
  max_eq_left hq


theorem _root_.Rat.le_coe_toNNRat (q : ℚ) : q ≤ q.toNNRat :=
  le_max_left _ _


@[simp]
theorem coe_nonneg (q : ℚ≥0) : (0 : ℚ) ≤ q :=
  q.2


@[simp, norm_cast] lemma coe_zero : ((0 : ℚ≥0) : ℚ) = 0 := rfl

@[simp] lemma num_zero : num 0 = 0 := rfl

@[simp] lemma den_zero : den 0 = 1 := rfl


@[simp, norm_cast] lemma coe_one : ((1 : ℚ≥0) : ℚ) = 1 := rfl

@[simp] lemma num_one : num 1 = 1 := rfl

@[simp] lemma den_one : den 1 = 1 := rfl


@[simp, norm_cast]
theorem coe_add (p q : ℚ≥0) : ((p + q : ℚ≥0) : ℚ) = p + q :=
  rfl


@[simp, norm_cast]
theorem coe_mul (p q : ℚ≥0) : ((p * q : ℚ≥0) : ℚ) = p * q :=
  rfl


@[simp, norm_cast] lemma coe_pow (q : ℚ≥0) (n : ℕ) : (↑(q ^ n) : ℚ) = (q : ℚ) ^ n :=
  rfl


                                                                        /-
                                                                          q : NNRat
                                                                          n : Nat
                                                                          ⊢ Eq (HPow.hPow q n).num (HPow.hPow q.num n)
                                                                        -/
@[simp] lemma num_pow (q : ℚ≥0) (n : ℕ) : (q ^ n).num = q.num ^ n := by simp [num, Int.natAbs_pow]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

@[simp] lemma den_pow (q : ℚ≥0) (n : ℕ) : (q ^ n).den = q.den ^ n := rfl


@[simp, norm_cast]
theorem coe_sub (h : q ≤ p) : ((p - q : ℚ≥0) : ℚ) = p - q :=
                                     /-
                                       p q : NNRat
                                       h : LE.le q p
                                       ⊢ LE.le (↑q) (HSub.hSub (↑p) 0)
                                     -/
  max_eq_left <| le_sub_comm.2 <| by rwa [sub_zero]
                                     /-
                                       🎉 no goals
                                     -/

-- See note [specialised high priority simp lemma]

@[simp high]
                                                /-
                                                  q : NNRat
                                                  ⊢ Iff (Eq (↑q) 0) (Eq q 0)
                                                -/
theorem coe_eq_zero : (q : ℚ) = 0 ↔ q = 0 := by norm_cast
                                                /-
                                                  🎉 no goals
                                                -/


theorem coe_ne_zero : (q : ℚ) ≠ 0 ↔ q ≠ 0 :=
  coe_eq_zero.not


@[norm_cast]
theorem coe_le_coe : (p : ℚ) ≤ q ↔ p ≤ q :=
  Iff.rfl


@[norm_cast]
theorem coe_lt_coe : (p : ℚ) < q ↔ p < q :=
  Iff.rfl

-- `cast_pos`, defined in a later file, makes this lemma redundant

@[simp, norm_cast, nolint simpNF]
theorem coe_pos : (0 : ℚ) < q ↔ 0 < q :=
  Iff.rfl


theorem coe_mono : Monotone ((↑) : ℚ≥0 → ℚ) :=
  fun _ _ ↦ coe_le_coe.2


theorem toNNRat_mono : Monotone toNNRat :=
  fun _ _ h ↦ max_le_max h le_rfl


@[simp]
theorem toNNRat_coe (q : ℚ≥0) : toNNRat q = q :=
  ext <| max_eq_left q.2


@[simp]
theorem toNNRat_coe_nat (n : ℕ) : toNNRat n = n :=
            /-
              n : Nat
              ⊢ Eq ↑(↑n).toNNRat ↑↑n
            -/
  ext <| by simp only [Nat.cast_nonneg', Rat.coe_toNNRat]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- `toNNRat` and `(↑) : ℚ≥0 → ℚ` form a Galois insertion. -/
protected def gi : GaloisInsertion toNNRat (↑) :=
  GaloisInsertion.monotoneIntro coe_mono toNNRat_mono Rat.le_coe_toNNRat toNNRat_coe


/-- Coercion `ℚ≥0 → ℚ` as a `RingHom`. -/
def coeHom : ℚ≥0 →+* ℚ where
  toFun := (↑)
  map_one' := coe_one
  map_mul' := coe_mul
  map_zero' := coe_zero
  map_add' := coe_add


@[simp, norm_cast] lemma coe_natCast (n : ℕ) : (↑(↑n : ℚ≥0) : ℚ) = n := rfl

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem mk_natCast (n : ℕ) : @Eq ℚ≥0 (⟨(n : ℚ), Nat.cast_nonneg' n⟩ : ℚ≥0) n :=
  rfl


@[deprecated (since := "2024-04-05")] alias mk_coe_nat := mk_natCast


@[simp]
theorem coe_coeHom : ⇑coeHom = ((↑) : ℚ≥0 → ℚ) :=
  rfl


@[norm_cast]
theorem nsmul_coe (q : ℚ≥0) (n : ℕ) : ↑(n • q) = n • (q : ℚ) :=
  coeHom.toAddMonoidHom.map_nsmul _ _


theorem bddAbove_coe {s : Set ℚ≥0} : BddAbove ((↑) '' s : Set ℚ) ↔ BddAbove s :=
  ⟨fun ⟨b, hb⟩ ↦
    ⟨toNNRat b, fun ⟨y, _⟩ hys ↦
      show y ≤ max b 0 from (hb <| Set.mem_image_of_mem _ hys).trans <| le_max_left _ _⟩,
    fun ⟨b, hb⟩ ↦ ⟨b, fun _ ⟨_, hx, Eq⟩ ↦ Eq ▸ hb hx⟩⟩


theorem bddBelow_coe (s : Set ℚ≥0) : BddBelow (((↑) : ℚ≥0 → ℚ) '' s) :=
  ⟨0, fun _ ⟨q, _, h⟩ ↦ h ▸ q.2⟩

-- `cast_max`, defined in a later file, makes this lemma redundant

@[simp, norm_cast, nolint simpNF]
theorem coe_max (x y : ℚ≥0) : ((max x y : ℚ≥0) : ℚ) = max (x : ℚ) (y : ℚ) :=
  coe_mono.map_max

-- `cast_max`, defined in a later file, makes this lemma redundant

@[simp, norm_cast, nolint simpNF]
theorem coe_min (x y : ℚ≥0) : ((min x y : ℚ≥0) : ℚ) = min (x : ℚ) (y : ℚ) :=
  coe_mono.map_min


theorem sub_def (p q : ℚ≥0) : p - q = toNNRat (p - q) :=
  rfl


@[simp]
theorem abs_coe (q : ℚ≥0) : |(q : ℚ)| = q :=
  abs_of_nonneg q.2

-- See note [specialised high priority simp lemma]

@[simp high]
theorem nonpos_iff_eq_zero (q : ℚ≥0) : q ≤ 0 ↔ q = 0 :=
  ⟨fun h => le_antisymm h q.2, fun h => h.symm ▸ q.2⟩


@[simp]
theorem toNNRat_zero : toNNRat 0 = 0 := rfl


@[simp]
theorem toNNRat_one : toNNRat 1 = 1 := rfl


@[simp]
                                                  /-
                                                    q : Rat
                                                    ⊢ Iff (LT.lt 0 q.toNNRat) (LT.lt 0 q)
                                                  -/
theorem toNNRat_pos : 0 < toNNRat q ↔ 0 < q := by simp [toNNRat, ← coe_lt_coe]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem toNNRat_eq_zero : toNNRat q = 0 ↔ q ≤ 0 := by
  /-
    q : Rat
    ⊢ Iff (Eq q.toNNRat 0) (LE.le q 0)
  -/
  simpa [-toNNRat_pos] using (@toNNRat_pos q).not
  /-
    🎉 no goals
  -/


alias ⟨_, toNNRat_of_nonpos⟩ := toNNRat_eq_zero


@[simp]
theorem toNNRat_le_toNNRat_iff (hp : 0 ≤ p) : toNNRat q ≤ toNNRat p ↔ q ≤ p := by
  /-
    p q : Rat
    hp : LE.le 0 p
    ⊢ Iff (LE.le q.toNNRat p.toNNRat) (LE.le q p)
  -/
  simp [← coe_le_coe, toNNRat, hp]
  /-
    🎉 no goals
  -/


@[simp]
theorem toNNRat_lt_toNNRat_iff' : toNNRat q < toNNRat p ↔ q < p ∧ 0 < p := by
  /-
    p q : Rat
    ⊢ Iff (LT.lt q.toNNRat p.toNNRat) (And (LT.lt q p) (LT.lt 0 p))
  -/
  simp [← coe_lt_coe, toNNRat, lt_irrefl]
  /-
    🎉 no goals
  -/


theorem toNNRat_lt_toNNRat_iff (h : 0 < p) : toNNRat q < toNNRat p ↔ q < p :=
  toNNRat_lt_toNNRat_iff'.trans (and_iff_left h)


theorem toNNRat_lt_toNNRat_iff_of_nonneg (hq : 0 ≤ q) : toNNRat q < toNNRat p ↔ q < p :=
  toNNRat_lt_toNNRat_iff'.trans ⟨And.left, fun h ↦ ⟨h, hq.trans_lt h⟩⟩


@[simp]
theorem toNNRat_add (hq : 0 ≤ q) (hp : 0 ≤ p) : toNNRat (q + p) = toNNRat q + toNNRat p :=
                  /-
                    p q : Rat
                    hq : LE.le 0 q
                    hp : LE.le 0 p
                    ⊢ Eq ↑(HAdd.hAdd q p).toNNRat ↑(HAdd.hAdd q.toNNRat p.toNNRat)
                  -/
  NNRat.ext <| by simp [toNNRat, hq, hp, add_nonneg]
                  /-
                    🎉 no goals
                  -/


theorem toNNRat_add_le : toNNRat (q + p) ≤ toNNRat q + toNNRat p :=
  coe_le_coe.1 <| max_le (add_le_add (le_max_left _ _) (le_max_left _ _)) <| coe_nonneg _


theorem toNNRat_le_iff_le_coe {p : ℚ≥0} : toNNRat q ≤ p ↔ q ≤ ↑p :=
  NNRat.gi.gc q p


theorem le_toNNRat_iff_coe_le {q : ℚ≥0} (hp : 0 ≤ p) : q ≤ toNNRat p ↔ ↑q ≤ p := by
  /-
    p : Rat
    q : NNRat
    hp : LE.le 0 p
    ⊢ Iff (LE.le q p.toNNRat) (LE.le (↑q) p)
  -/
  rw [← coe_le_coe, Rat.coe_toNNRat p hp]
  /-
    🎉 no goals
  -/


theorem le_toNNRat_iff_coe_le' {q : ℚ≥0} (hq : 0 < q) : q ≤ toNNRat p ↔ ↑q ≤ p :=
  (le_or_lt 0 p).elim le_toNNRat_iff_coe_le fun hp ↦ by
    /-
      p : Rat
      q : NNRat
      hq : LT.lt 0 q
      hp : LT.lt p 0
      ⊢ Iff (LE.le q p.toNNRat) (LE.le (↑q) p)
    -/
    simp only [(hp.trans_le q.coe_nonneg).not_le, toNNRat_eq_zero.2 hp.le, hq.not_le]
    /-
      🎉 no goals
    -/


theorem toNNRat_lt_iff_lt_coe {p : ℚ≥0} (hq : 0 ≤ q) : toNNRat q < p ↔ q < ↑p := by
  /-
    q : Rat
    p : NNRat
    hq : LE.le 0 q
    ⊢ Iff (LT.lt q.toNNRat p) (LT.lt q ↑p)
  -/
  rw [← coe_lt_coe, Rat.coe_toNNRat q hq]
  /-
    🎉 no goals
  -/


theorem lt_toNNRat_iff_coe_lt {q : ℚ≥0} : q < toNNRat p ↔ ↑q < p :=
  NNRat.gi.gc.lt_iff_lt


theorem toNNRat_mul (hp : 0 ≤ p) : toNNRat (p * q) = toNNRat p * toNNRat q := by
  /-
    p q : Rat
    hp : LE.le 0 p
    ⊢ Eq (HMul.hMul p q).toNNRat (HMul.hMul p.toNNRat q.toNNRat)
  -/
  rcases le_total 0 q with hq | hq
    /-
      case inl
      p q : Rat
      hp : LE.le 0 p
      hq : LE.le 0 q
      ⊢ Eq (HMul.hMul p q).toNNRat (HMul.hMul p.toNNRat q.toNNRat)
    -/
  · ext; simp [toNNRat, hp, hq, max_eq_left, mul_nonneg]
         /-
           🎉 no goals
         -/
    /-
      case inr
      p q : Rat
      hp : LE.le 0 p
      hq : LE.le q 0
      ⊢ Eq (HMul.hMul p q).toNNRat (HMul.hMul p.toNNRat q.toNNRat)
    -/
  · have hpq := mul_nonpos_of_nonneg_of_nonpos hp hq
    /-
      case inr
      p q : Rat
      hp : LE.le 0 p
      hq : LE.le q 0
      hpq : LE.le (HMul.hMul p q) 0
      ⊢ Eq (HMul.hMul p q).toNNRat (HMul.hMul p.toNNRat q.toNNRat)
    -/
    rw [toNNRat_eq_zero.2 hq, toNNRat_eq_zero.2 hpq, mul_zero]
    /-
      🎉 no goals
    -/


/-- The absolute value on `ℚ` as a map to `ℚ≥0`. -/
@[pp_nodot]
def Rat.nnabs (x : ℚ) : ℚ≥0 :=
  ⟨abs x, abs_nonneg x⟩


@[norm_cast, simp]
theorem Rat.coe_nnabs (x : ℚ) : (Rat.nnabs x : ℚ) = abs x := rfl


@[norm_cast] lemma num_coe (q : ℚ≥0) : (q : ℚ).num = q.num := by
  /-
    q : NNRat
    ⊢ Eq (↑q).num ↑q.num
  -/
  simp only [num, Int.natCast_natAbs, Rat.num_nonneg, coe_nonneg, abs_of_nonneg]
  /-
    🎉 no goals
  -/


theorem natAbs_num_coe : (q : ℚ).num.natAbs = q.num := rfl


@[norm_cast] lemma den_coe : (q : ℚ).den = q.den := rfl


                                                    /-
                                                      q : NNRat
                                                      ⊢ Iff (Ne q.num 0) (Ne q 0)
                                                    -/
@[simp] lemma num_ne_zero : q.num ≠ 0 ↔ q ≠ 0 := by simp [num]
                                                    /-
                                                      🎉 no goals
                                                    -/

@[simp] lemma num_pos : 0 < q.num ↔ 0 < q := by
  /-
    q : NNRat
    ⊢ Iff (LT.lt 0 q.num) (LT.lt 0 q)
  -/
  simpa [num, -nonpos_iff_eq_zero] using nonpos_iff_eq_zero _ |>.not.symm
  /-
    🎉 no goals
  -/

@[simp] lemma den_pos (q : ℚ≥0) : 0 < q.den := Rat.den_pos _

@[simp] lemma den_ne_zero (q : ℚ≥0) : q.den ≠ 0 := Rat.den_ne_zero _


                                                            /-
                                                              q : NNRat
                                                              ⊢ q.num.Coprime q.den
                                                            -/
lemma coprime_num_den (q : ℚ≥0) : q.num.Coprime q.den := by simpa [num, den] using Rat.reduced _
                                                            /-
                                                              🎉 no goals
                                                            -/

-- TODO: Rename `Rat.coe_nat_num`, `Rat.intCast_den`, `Rat.ofNat_num`, `Rat.ofNat_den`

@[simp, norm_cast] lemma num_natCast (n : ℕ) : num n = n := rfl

@[simp, norm_cast] lemma den_natCast (n : ℕ) : den n = 1 := rfl

-- See note [no_index around OfNat.ofNat]

@[simp] lemma num_ofNat (n : ℕ) [n.AtLeastTwo] : num (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  rfl

@[simp] lemma den_ofNat (n : ℕ) [n.AtLeastTwo] : den (no_index (OfNat.ofNat n)) = 1 := rfl


theorem ext_num_den (hn : p.num = q.num) (hd : p.den = q.den) : p = q := by
  /-
    p q : NNRat
    hn : Eq p.num q.num
    hd : Eq p.den q.den
    ⊢ Eq p q
  -/
  refine ext <| Rat.ext ?_ hd
  /-
    p q : NNRat
    hn : Eq p.num q.num
    hd : Eq p.den q.den
    ⊢ Eq (↑p).num (↑q).num
  -/
  simpa [num_coe]
  /-
    🎉 no goals
  -/


theorem ext_num_den_iff : p = q ↔ p.num = q.num ∧ p.den = q.den :=
      /-
        p q : NNRat
        ⊢ Eq p q → And (Eq p.num q.num) (Eq p.den q.den)
      -/
  ⟨by rintro rfl; exact ⟨rfl, rfl⟩, fun h ↦ ext_num_den h.1 h.2⟩
                  /-
                    🎉 no goals
                  -/


/-- Form the quotient `n / d` where `n d : ℕ`.

See also `Rat.divInt` and `mkRat`. -/
def divNat (n d : ℕ) : ℚ≥0 :=
  ⟨.divInt n d, Rat.divInt_nonneg (Int.ofNat_zero_le n) (Int.ofNat_zero_le d)⟩


@[simp, norm_cast] lemma coe_divNat (n d : ℕ) : (divNat n d : ℚ) = .divInt n d := rfl


lemma mk_divInt (n d : ℕ) :
    ⟨.divInt n d, Rat.divInt_nonneg (Int.ofNat_zero_le n) (Int.ofNat_zero_le d)⟩ = divNat n d := rfl


lemma divNat_inj (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) : divNat n₁ d₁ = divNat n₂ d₂ ↔ n₁ * d₂ = n₂ * d₁ := by
  /-
    n₁ n₂ d₁ d₂ : Nat
    h₁ : Ne d₁ 0
    h₂ : Ne d₂ 0
    ⊢ Iff (Eq (NNRat.divNat n₁ d₁) (NNRat.divNat n₂ d₂)) (Eq (HMul.hMul n₁ d₂) (HM …
  -/
  rw [← coe_inj]; simp [Rat.mkRat_eq_iff, h₁, h₂]; norm_cast
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                         /-
                                                           n : Nat
                                                           ⊢ Eq (NNRat.divNat n 0) 0
                                                         -/
@[simp] lemma divNat_zero (n : ℕ) : divNat n 0 = 0 := by simp [divNat]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp] lemma num_divNat_den (q : ℚ≥0) : divNat q.num q.den = q :=
            /-
              q : NNRat
              ⊢ Eq ↑(NNRat.divNat q.num q.den) ↑q
            -/
  ext <| by rw [← (q : ℚ).mkRat_num_den']; simp [num_coe, den_coe]
                                           /-
                                             🎉 no goals
                                           -/


lemma natCast_eq_divNat (n : ℕ) : (n : ℚ≥0) = divNat n 1 := (num_divNat_den _).symm


lemma divNat_mul_divNat (n₁ n₂ : ℕ) {d₁ d₂} (hd₁ : d₁ ≠ 0) (hd₂ : d₂ ≠ 0) :
    divNat n₁ d₁ * divNat n₂ d₂ = divNat (n₁ * n₂) (d₁ * d₂) := by
  /-
    n₁ n₂ d₁ d₂ : Nat
    hd₁ : Ne d₁ 0
    hd₂ : Ne d₂ 0
    ⊢ Eq (HMul.hMul (NNRat.divNat n₁ d₁) (NNRat.divNat n₂ d₂)) (NNRat.divNat (HMul …
  -/
  ext; push_cast; exact Rat.divInt_mul_divInt _ _ (mod_cast hd₁) (mod_cast hd₂)
                  /-
                    🎉 no goals
                  -/


lemma divNat_mul_left {a : ℕ} (ha : a ≠ 0) (n d : ℕ) : divNat (a * n) (a * d) = divNat n d := by
  /-
    a : Nat
    ha : Ne a 0
    n d : Nat
    ⊢ Eq (NNRat.divNat (HMul.hMul a n) (HMul.hMul a d)) (NNRat.divNat n d)
  -/
  ext; push_cast; exact Rat.divInt_mul_left (mod_cast ha)
                  /-
                    🎉 no goals
                  -/


lemma divNat_mul_right {a : ℕ} (ha : a ≠ 0) (n d : ℕ) : divNat (n * a) (d * a) = divNat n d := by
  /-
    a : Nat
    ha : Ne a 0
    n d : Nat
    ⊢ Eq (NNRat.divNat (HMul.hMul n a) (HMul.hMul d a)) (NNRat.divNat n d)
  -/
  ext; push_cast; exact Rat.divInt_mul_right (mod_cast ha)
                  /-
                    🎉 no goals
                  -/


@[simp] lemma mul_den_eq_num (q : ℚ≥0) : q * q.den = q.num := by
  /-
    q : NNRat
    ⊢ Eq (HMul.hMul q ↑q.den) ↑q.num
  -/
  ext
  /-
    case a
    q : NNRat
    ⊢ Eq ↑(HMul.hMul q ↑q.den) ↑↑q.num
  -/
  push_cast
  /-
    case a
    q : NNRat
    ⊢ Eq (HMul.hMul ↑q ↑q.den) ↑q.num
  -/
  rw [← Int.cast_natCast, ← den_coe, ← Int.cast_natCast q.num, ← num_coe]
  /-
    case a
    q : NNRat
    ⊢ Eq (HMul.hMul ↑q ↑↑(↑q).den) ↑(↑q).num
  -/
  exact Rat.mul_den_eq_num _
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   q : NNRat
                                                                   ⊢ Eq (HMul.hMul (↑q.den) q) ↑q.num
                                                                 -/
@[simp] lemma den_mul_eq_num (q : ℚ≥0) : q.den * q = q.num := by rw [mul_comm, mul_den_eq_num]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Define a (dependent) function or prove `∀ r : ℚ, p r` by dealing with nonnegative rational
numbers of the form `n / d` with `d ≠ 0` and `n`, `d` coprime. -/
@[elab_as_elim]
def numDenCasesOn.{u} {C : ℚ≥0 → Sort u} (q) (H : ∀ n d, d ≠ 0 → n.Coprime d → C (divNat n d)) :
              /-
                p q✝ : NNRat
                n₁ n₂ d₁ d₂ : Nat
                C : NNRat → Sort u
                q : NNRat
                H : (n d : Nat) → Ne d 0 → n.Coprime d → C (NNRat.divNat n d)
                ⊢ C q
              -/
    C q := by rw [← q.num_divNat_den]; exact H _ _ q.den_ne_zero q.coprime_num_den
                                       /-
                                         🎉 no goals
                                       -/


lemma add_def (q r : ℚ≥0) : q + r = divNat (q.num * r.den + r.num * q.den) (q.den * r.den) := by
  /-
    q r : NNRat
    ⊢ Eq (HAdd.hAdd q r) (NNRat.divNat (HAdd.hAdd (HMul.hMul q.num r.den) (HMul.hM …
  -/
  ext; simp [Rat.add_def', Rat.mkRat_eq_divInt, num_coe, den_coe]
       /-
         🎉 no goals
       -/


lemma mul_def (q r : ℚ≥0) : q * r = divNat (q.num * r.num) (q.den * r.den) := by
  /-
    q r : NNRat
    ⊢ Eq (HMul.hMul q r) (NNRat.divNat (HMul.hMul q.num r.num) (HMul.hMul q.den r. …
  -/
  ext; simp [Rat.mul_eq_mkRat, Rat.mkRat_eq_divInt, num_coe, den_coe]
       /-
         🎉 no goals
       -/


theorem lt_def {p q : ℚ≥0} : p < q ↔ p.num * q.den < q.num * p.den := by
  /-
    p q : NNRat
    ⊢ Iff (LT.lt p q) (LT.lt (HMul.hMul p.num q.den) (HMul.hMul q.num p.den))
  -/
  rw [← NNRat.coe_lt_coe, Rat.lt_def]; norm_cast
                                       /-
                                         🎉 no goals
                                       -/


theorem le_def {p q : ℚ≥0} : p ≤ q ↔ p.num * q.den ≤ q.num * p.den := by
  /-
    p q : NNRat
    ⊢ Iff (LE.le p q) (LE.le (HMul.hMul p.num q.den) (HMul.hMul q.num p.den))
  -/
  rw [← NNRat.coe_le_coe, Rat.le_def]; norm_cast
                                       /-
                                         🎉 no goals
                                       -/


@[qify_simps] lemma nnratCast_eq (a b : ℚ≥0) : a = b ↔ (a : ℚ) = (b : ℚ) := NNRat.coe_inj.symm

@[qify_simps] lemma nnratCast_le (a b : ℚ≥0) : a ≤ b ↔ (a : ℚ) ≤ (b : ℚ) := NNRat.coe_le_coe.symm

@[qify_simps] lemma nnratCast_lt (a b : ℚ≥0) : a < b ↔ (a : ℚ) < (b : ℚ) := NNRat.coe_lt_coe.symm

@[qify_simps] lemma nnratCast_ne (a b : ℚ≥0) : a ≠ b ↔ (a : ℚ) ≠ (b : ℚ) := NNRat.ne_iff.symm


