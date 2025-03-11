/-- Coercion `ℕ∞ → Cardinal`. It sends natural numbers to natural numbers and `⊤` to `ℵ₀`.

See also `Cardinal.ofENatHom` for a bundled homomorphism version. -/
@[coe] def ofENat : ℕ∞ → Cardinal
  | (n : ℕ) => n
  | ⊤ => ℵ₀


instance : Coe ENat Cardinal := ⟨Cardinal.ofENat⟩


@[simp, norm_cast] lemma ofENat_top : ofENat ⊤ = ℵ₀ := rfl

@[simp, norm_cast] lemma ofENat_nat (n : ℕ) : ofENat n = n := rfl

@[simp, norm_cast] lemma ofENat_zero : ofENat 0 = 0 := rfl

@[simp, norm_cast] lemma ofENat_one : ofENat 1 = 1 := rfl


@[simp, norm_cast] lemma ofENat_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((no_index (OfNat.ofNat n : ℕ∞)) : Cardinal) = OfNat.ofNat n :=
  rfl


lemma ofENat_strictMono : StrictMono ofENat :=
  WithTop.strictMono_iff.2 ⟨Nat.strictMono_cast, nat_lt_aleph0⟩


@[simp, norm_cast]
lemma ofENat_lt_ofENat {m n : ℕ∞} : (m : Cardinal) < n ↔ m < n :=
  ofENat_strictMono.lt_iff_lt


@[gcongr, mono] alias ⟨_, ofENat_lt_ofENat_of_lt⟩ := ofENat_lt_ofENat


@[simp, norm_cast]
lemma ofENat_lt_aleph0 {m : ℕ∞} : (m : Cardinal) < ℵ₀ ↔ m < ⊤ :=
  ofENat_lt_ofENat (n := ⊤)


                                                                          /-
                                                                            m : ENat
                                                                            n : Nat
                                                                            ⊢ Iff (LT.lt ↑m ↑n) (LT.lt m ↑n)
                                                                          -/
@[simp] lemma ofENat_lt_nat {m : ℕ∞} {n : ℕ} : ofENat m < n ↔ m < n := by norm_cast
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp] lemma ofENat_lt_ofNat {m : ℕ∞} {n : ℕ} [n.AtLeastTwo] :
    ofENat m < no_index (OfNat.ofNat n) ↔ m < OfNat.ofNat n := ofENat_lt_nat


                                                                                /-
                                                                                  m : Nat
                                                                                  n : ENat
                                                                                  ⊢ Iff (LT.lt ↑m ↑n) (LT.lt (↑m) n)
                                                                                -/
@[simp] lemma nat_lt_ofENat {m : ℕ} {n : ℕ∞} : (m : Cardinal) < n ↔ m < n := by norm_cast
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                     /-
                                                                       m : ENat
                                                                       ⊢ Iff (LT.lt 0 ↑m) (LT.lt 0 m)
                                                                     -/
@[simp] lemma ofENat_pos {m : ℕ∞} : 0 < (m : Cardinal) ↔ 0 < m := by norm_cast
                                                                     /-
                                                                       🎉 no goals
                                                                     -/

                                                                        /-
                                                                          m : ENat
                                                                          ⊢ Iff (LT.lt 1 ↑m) (LT.lt 1 m)
                                                                        -/
@[simp] lemma one_lt_ofENat {m : ℕ∞} : 1 < (m : Cardinal) ↔ 1 < m := by norm_cast
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp, norm_cast] lemma ofNat_lt_ofENat {m : ℕ} [m.AtLeastTwo] {n : ℕ∞} :
  no_index (OfNat.ofNat m : Cardinal) < n ↔ OfNat.ofNat m < n := nat_lt_ofENat


lemma ofENat_mono : Monotone ofENat := ofENat_strictMono.monotone


@[simp, norm_cast]
lemma ofENat_le_ofENat {m n : ℕ∞} : (m : Cardinal) ≤ n ↔ m ≤ n := ofENat_strictMono.le_iff_le


@[gcongr, mono] alias ⟨_, ofENat_le_ofENat_of_le⟩ := ofENat_le_ofENat


@[simp] lemma ofENat_le_aleph0 (n : ℕ∞) : ↑n ≤ ℵ₀ := ofENat_le_ofENat.2 le_top

                                                                          /-
                                                                            m : ENat
                                                                            n : Nat
                                                                            ⊢ Iff (LE.le ↑m ↑n) (LE.le m ↑n)
                                                                          -/
@[simp] lemma ofENat_le_nat {m : ℕ∞} {n : ℕ} : ofENat m ≤ n ↔ m ≤ n := by norm_cast
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                  /-
                                                                    m : ENat
                                                                    ⊢ Iff (LE.le (↑m) 1) (LE.le m 1)
                                                                  -/
@[simp] lemma ofENat_le_one {m : ℕ∞} : ofENat m ≤ 1 ↔ m ≤ 1 := by norm_cast
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp] lemma ofENat_le_ofNat {m : ℕ∞} {n : ℕ} [n.AtLeastTwo] :
    ofENat m ≤ no_index (OfNat.ofNat n) ↔ m ≤ OfNat.ofNat n := ofENat_le_nat


                                                                                /-
                                                                                  m : Nat
                                                                                  n : ENat
                                                                                  ⊢ Iff (LE.le ↑m ↑n) (LE.le (↑m) n)
                                                                                -/
@[simp] lemma nat_le_ofENat {m : ℕ} {n : ℕ∞} : (m : Cardinal) ≤ n ↔ m ≤ n := by norm_cast
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                        /-
                                                                          n : ENat
                                                                          ⊢ Iff (LE.le 1 ↑n) (LE.le 1 n)
                                                                        -/
@[simp] lemma one_le_ofENat {n : ℕ∞} : 1 ≤ (n : Cardinal) ↔ 1 ≤ n := by norm_cast
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
lemma ofNat_le_ofENat {m : ℕ} [m.AtLeastTwo] {n : ℕ∞} :
    no_index (OfNat.ofNat m : Cardinal) ≤ n ↔ OfNat.ofNat m ≤ n := nat_le_ofENat


lemma ofENat_injective : Injective ofENat := ofENat_strictMono.injective


@[simp, norm_cast]
lemma ofENat_inj {m n : ℕ∞} : (m : Cardinal) = n ↔ m = n := ofENat_injective.eq_iff


                                                                                /-
                                                                                  m : ENat
                                                                                  n : Nat
                                                                                  ⊢ Iff (Eq ↑m ↑n) (Eq m ↑n)
                                                                                -/
@[simp] lemma ofENat_eq_nat {m : ℕ∞} {n : ℕ} : (m : Cardinal) = n ↔ m = n := by norm_cast
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                                /-
                                                                                  m : Nat
                                                                                  n : ENat
                                                                                  ⊢ Iff (Eq ↑m ↑n) (Eq (↑m) n)
                                                                                -/
@[simp] lemma nat_eq_ofENat {m : ℕ} {n : ℕ∞} : (m : Cardinal) = n ↔ m = n := by norm_cast
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


                                                                         /-
                                                                           m : ENat
                                                                           ⊢ Iff (Eq (↑m) 0) (Eq m 0)
                                                                         -/
@[simp] lemma ofENat_eq_zero {m : ℕ∞} : (m : Cardinal) = 0 ↔ m = 0 := by norm_cast
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

                                                                         /-
                                                                           m : ENat
                                                                           ⊢ Iff (Eq 0 ↑m) (Eq m 0)
                                                                         -/
@[simp] lemma zero_eq_ofENat {m : ℕ∞} : 0 = (m : Cardinal) ↔ m = 0 := by norm_cast; apply eq_comm
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


                                                                        /-
                                                                          m : ENat
                                                                          ⊢ Iff (Eq (↑m) 1) (Eq m 1)
                                                                        -/
@[simp] lemma ofENat_eq_one {m : ℕ∞} : (m : Cardinal) = 1 ↔ m = 1 := by norm_cast
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

                                                                        /-
                                                                          m : ENat
                                                                          ⊢ Iff (Eq 1 ↑m) (Eq m 1)
                                                                        -/
@[simp] lemma one_eq_ofENat {m : ℕ∞} : 1 = (m : Cardinal) ↔ m = 1 := by norm_cast; apply eq_comm
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp] lemma ofENat_eq_ofNat {m : ℕ∞} {n : ℕ} [n.AtLeastTwo] :
    (m : Cardinal) = no_index (OfNat.ofNat n) ↔ m = OfNat.ofNat n := ofENat_eq_nat


@[simp] lemma ofNat_eq_ofENat {m : ℕ} {n : ℕ∞} [m.AtLeastTwo] :
    no_index (OfNat.ofNat m) = (n : Cardinal) ↔ OfNat.ofNat m = n := nat_eq_ofENat


@[simp, norm_cast] lemma lift_ofENat : ∀ m : ℕ∞, lift.{u, v} m = m
  | (m : ℕ) => lift_natCast m
  | ⊤ => lift_aleph0


@[simp] lemma lift_lt_ofENat {x : Cardinal.{v}} {m : ℕ∞} : lift.{u} x < m ↔ x < m := by
  /-
    x : Cardinal.{v}
    m : ENat
    ⊢ Iff (LT.lt (Cardinal.lift.{u, v} x) ↑m) (LT.lt x ↑m)
  -/
  rw [← lift_ofENat.{u, v}, lift_lt]
  /-
    🎉 no goals
  -/


@[simp] lemma lift_le_ofENat {x : Cardinal.{v}} {m : ℕ∞} : lift.{u} x ≤ m ↔ x ≤ m := by
  /-
    x : Cardinal.{v}
    m : ENat
    ⊢ Iff (LE.le (Cardinal.lift.{u, v} x) ↑m) (LE.le x ↑m)
  -/
  rw [← lift_ofENat.{u, v}, lift_le]
  /-
    🎉 no goals
  -/


@[simp] lemma lift_eq_ofENat {x : Cardinal.{v}} {m : ℕ∞} : lift.{u} x = m ↔ x = m := by
  /-
    x : Cardinal.{v}
    m : ENat
    ⊢ Iff (Eq (Cardinal.lift.{u, v} x) ↑m) (Eq x ↑m)
  -/
  rw [← lift_ofENat.{u, v}, lift_inj]
  /-
    🎉 no goals
  -/


@[simp] lemma ofENat_lt_lift {x : Cardinal.{v}} {m : ℕ∞} : m < lift.{u} x ↔ m < x := by
  /-
    x : Cardinal.{v}
    m : ENat
    ⊢ Iff (LT.lt (↑m) (Cardinal.lift.{u, v} x)) (LT.lt (↑m) x)
  -/
  rw [← lift_ofENat.{u, v}, lift_lt]
  /-
    🎉 no goals
  -/


@[simp] lemma ofENat_le_lift {x : Cardinal.{v}} {m : ℕ∞} : m ≤ lift.{u} x ↔ m ≤ x := by
  /-
    x : Cardinal.{v}
    m : ENat
    ⊢ Iff (LE.le (↑m) (Cardinal.lift.{u, v} x)) (LE.le (↑m) x)
  -/
  rw [← lift_ofENat.{u, v}, lift_le]
  /-
    🎉 no goals
  -/


@[simp] lemma ofENat_eq_lift {x : Cardinal.{v}} {m : ℕ∞} : m = lift.{u} x ↔ m = x := by
  /-
    x : Cardinal.{v}
    m : ENat
    ⊢ Iff (Eq (↑m) (Cardinal.lift.{u, v} x)) (Eq (↑m) x)
  -/
  rw [← lift_ofENat.{u, v}, lift_inj]
  /-
    🎉 no goals
  -/


@[simp]
lemma range_ofENat : range ofENat = Iic ℵ₀ := by
  /-
    ⊢ Eq (Set.range Cardinal.ofENat) (Set.Iic Cardinal.aleph0)
  -/
  refine (range_subset_iff.2 ofENat_le_aleph0).antisymm fun x (hx : x ≤ ℵ₀) ↦ ?_
  /-
    x : Cardinal.{u_1}
    hx : LE.le x Cardinal.aleph0
    ⊢ Membership.mem (Set.range Cardinal.ofENat) x
  -/
  rcases hx.lt_or_eq with hlt | rfl
    /-
      case inl
      x : Cardinal.{u_1}
      hx : LE.le x Cardinal.aleph0
      hlt : LT.lt x Cardinal.aleph0
      ⊢ Membership.mem (Set.range Cardinal.ofENat) x
    -/
  · lift x to ℕ using hlt
    /-
      case inl.intro
      x : Nat
      hx : LE.le (↑x) Cardinal.aleph0
      ⊢ Membership.mem (Set.range Cardinal.ofENat) ↑x
    -/
    exact mem_range_self (x : ℕ∞)
    /-
      🎉 no goals
    -/
    /-
      case inr
      hx : LE.le Cardinal.aleph0 Cardinal.aleph0
      ⊢ Membership.mem (Set.range Cardinal.ofENat) Cardinal.aleph0
    -/
  · exact mem_range_self (⊤ : ℕ∞)
    /-
      🎉 no goals
    -/


instance : CanLift Cardinal ℕ∞ (↑) (· ≤ ℵ₀) where
  prf x := (Set.ext_iff.1 range_ofENat x).2


/-- Unbundled version of `Cardinal.toENat`. -/
noncomputable def toENatAux : Cardinal.{u} → ℕ∞ := extend Nat.cast Nat.cast fun _ ↦ ⊤


lemma toENatAux_nat (n : ℕ) : toENatAux n = n := Nat.cast_injective.extend_apply ..

lemma toENatAux_zero : toENatAux 0 = 0 := toENatAux_nat 0


lemma toENatAux_eq_top {a : Cardinal} (ha : ℵ₀ ≤ a) : toENatAux a = ⊤ :=
  extend_apply' _ _ _ fun ⟨n, hn⟩ ↦ ha.not_lt <| hn ▸ nat_lt_aleph0 n


lemma toENatAux_ofENat : ∀ n : ℕ∞, toENatAux n = n
  | (n : ℕ) => toENatAux_nat n
  | ⊤ => toENatAux_eq_top le_rfl


lemma toENatAux_gc : GaloisConnection (↑) toENatAux := fun n x ↦ by
  cases lt_or_le x ℵ₀ with
  | inl hx => lift x to ℕ using hx; simp
  | inr hx => simp [toENatAux_eq_top hx, (ofENat_le_aleph0 n).trans hx]


theorem toENatAux_le_nat {x : Cardinal} {n : ℕ} : toENatAux x ≤ n ↔ x ≤ n := by
  cases lt_or_le x ℵ₀ with
  | inl hx => lift x to ℕ using hx; simp
  | inr hx => simp [toENatAux_eq_top hx, (nat_lt_aleph0 n).trans_le hx]


lemma toENatAux_eq_nat {x : Cardinal} {n : ℕ} : toENatAux x = n ↔ x = n := by
  /-
    x : Cardinal.{u_1}
    n : Nat
    ⊢ Iff (Eq x.toENatAux ↑n) (Eq x ↑n)
  -/
  simp only [le_antisymm_iff, toENatAux_le_nat, ← toENatAux_gc _, ofENat_nat]
  /-
    🎉 no goals
  -/


lemma toENatAux_eq_zero {x : Cardinal} : toENatAux x = 0 ↔ x = 0 := toENatAux_eq_nat


/-- Projection from cardinals to `ℕ∞`. Sends all infinite cardinals to `⊤`.

We define this function as a bundled monotone ring homomorphism. -/
noncomputable def toENat : Cardinal.{u} →+*o ℕ∞ where
  toFun := toENatAux
  map_one' := toENatAux_nat 1
  map_mul' x y := by
    /-
      x y : Cardinal.{u}
      ⊢ Eq ({ toFun := Cardinal.toENatAux, map_one' := ⋯ }.toFun (HMul.hMul x y)) (H …
    -/
    wlog hle : x ≤ y; · rw [mul_comm, this y x (le_of_not_le hle), mul_comm]
                        /-
                          🎉 no goals
                        -/
    cases lt_or_le y ℵ₀ with
    | inl hy =>
      lift x to ℕ using hle.trans_lt hy; lift y to ℕ using hy
      simp only [← Nat.cast_mul, toENatAux_nat]
    | inr hy =>
      rcases eq_or_ne x 0 with rfl | hx
      · simp
      · simp only [toENatAux_eq_top hy]
        rw [toENatAux_eq_top, ENat.mul_top]
        · rwa [Ne, toENatAux_eq_zero]
        · exact le_mul_of_one_le_of_le (one_le_iff_ne_zero.2 hx) hy
  map_add' x y := by
    /-
      x y : Cardinal.{u}
      ⊢ Eq ((↑{ toFun := Cardinal.toENatAux, map_one' := ⋯, map_mul' := ⋯ }).toFun ( …
    -/
    wlog hle : x ≤ y; · rw [add_comm, this y x (le_of_not_le hle), add_comm]
                        /-
                          🎉 no goals
                        -/
    cases lt_or_le y ℵ₀ with
    | inl hy =>
      lift x to ℕ using hle.trans_lt hy; lift y to ℕ using hy
      simp only [← Nat.cast_add, toENatAux_nat]
    | inr hy =>
      simp only [toENatAux_eq_top hy, add_top]
      exact toENatAux_eq_top <| le_add_left hy
  map_zero' := toENatAux_zero
  monotone' := toENatAux_gc.monotone_u


/-- The coercion `Cardinal.ofENat` and the projection `Cardinal.toENat` form a Galois connection.
See also `Cardinal.gciENat`. -/
lemma enat_gc : GaloisConnection (↑) toENat := toENatAux_gc


@[simp] lemma toENat_ofENat (n : ℕ∞) : toENat n = n := toENatAux_ofENat n

@[simp] lemma toENat_comp_ofENat : toENat ∘ (↑) = id := funext toENat_ofENat


/-- The coercion `Cardinal.ofENat` and the projection `Cardinal.toENat`
form a Galois coinsertion. -/
noncomputable def gciENat : GaloisCoinsertion (↑) toENat :=
  enat_gc.toGaloisCoinsertion fun n ↦ (toENat_ofENat n).le


lemma toENat_strictMonoOn : StrictMonoOn toENat (Iic ℵ₀) := by
  /-
    ⊢ StrictMonoOn (⇑Cardinal.toENat) (Set.Iic Cardinal.aleph0)
  -/
  simp only [← range_ofENat, StrictMonoOn, forall_mem_range, toENat_ofENat, ofENat_lt_ofENat]
  /-
    ⊢ ∀ (i i_1 : ENat), LT.lt i i_1 → LT.lt i i_1
  -/
  exact fun _ _ ↦ id
  /-
    🎉 no goals
  -/


lemma toENat_injOn : InjOn toENat (Iic ℵ₀) := toENat_strictMonoOn.injOn


lemma ofENat_toENat_le (a : Cardinal) : ↑(toENat a) ≤ a := enat_gc.l_u_le _


@[simp]
lemma ofENat_toENat_eq_self {a : Cardinal} : toENat a = a ↔ a ≤ ℵ₀ := by
  /-
    a : Cardinal.{u_1}
    ⊢ Iff (Eq (↑(Cardinal.toENat a)) a) (LE.le a Cardinal.aleph0)
  -/
  rw [eq_comm, ← enat_gc.exists_eq_l]
  /-
    a : Cardinal.{u_1}
    ⊢ Iff (Exists fun a_1 => Eq a ↑a_1) (LE.le a Cardinal.aleph0)
  -/
  simpa only [mem_range, eq_comm] using Set.ext_iff.1 range_ofENat a
  /-
    🎉 no goals
  -/


@[simp] alias ⟨_, ofENat_toENat⟩ := ofENat_toENat_eq_self


lemma toENat_nat (n : ℕ) : toENat n = n := map_natCast _ n


@[simp] lemma toENat_le_nat {a : Cardinal} {n : ℕ} : toENat a ≤ n ↔ a ≤ n := toENatAux_le_nat

@[simp] lemma toENat_eq_nat {a : Cardinal} {n : ℕ} : toENat a = n ↔ a = n := toENatAux_eq_nat

@[simp] lemma toENat_eq_zero {a : Cardinal} : toENat a = 0 ↔ a = 0 := toENatAux_eq_zero

@[simp] lemma toENat_le_one {a : Cardinal} : toENat a ≤ 1 ↔ a ≤ 1 := toENat_le_nat

@[simp] lemma toENat_eq_one {a : Cardinal} : toENat a = 1 ↔ a = 1 := toENat_eq_nat


@[simp] lemma toENat_le_ofNat {a : Cardinal} {n : ℕ} [n.AtLeastTwo] :
    toENat a ≤ no_index (OfNat.ofNat n) ↔ a ≤ OfNat.ofNat n := toENat_le_nat


@[simp] lemma toENat_eq_ofNat {a : Cardinal} {n : ℕ} [n.AtLeastTwo] :
    toENat a = no_index (OfNat.ofNat n) ↔ a = OfNat.ofNat n := toENat_eq_nat


@[simp] lemma toENat_eq_top {a : Cardinal} : toENat a = ⊤ ↔ ℵ₀ ≤ a := enat_gc.u_eq_top


@[simp]
theorem toENat_lift {a : Cardinal.{v}} : toENat (lift.{u} a) = toENat a := by
  cases le_total a ℵ₀ with
  | inl ha => lift a to ℕ∞ using ha; simp
  | inr ha => simp [toENat_eq_top.2, ha]


theorem toENat_congr {α : Type u} {β : Type v} (e : α ≃ β) : toENat #α = toENat #β := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    ⊢ Eq (Cardinal.toENat (Cardinal.mk α)) (Cardinal.toENat (Cardinal.mk β))
  -/
  rw [← toENat_lift, lift_mk_eq.{_, _,v}.mpr ⟨e⟩, toENat_lift]
  /-
    🎉 no goals
  -/


lemma toENat_le_iff_of_le_aleph0 {c c' : Cardinal} (h : c ≤ ℵ₀) :
    toENat c ≤ toENat c' ↔ c ≤ c' := by
  /-
    c c' : Cardinal.{u_1}
    h : LE.le c Cardinal.aleph0
    ⊢ Iff (LE.le (Cardinal.toENat c) (Cardinal.toENat c')) (LE.le c c')
  -/
  lift c to ℕ∞ using h
  /-
    case intro
    c' : Cardinal.{u_1}
    c : ENat
    ⊢ Iff (LE.le (Cardinal.toENat ↑c) (Cardinal.toENat c')) (LE.le (↑c) c')
  -/
  simp_rw [toENat_ofENat, enat_gc _]
  /-
    🎉 no goals
  -/


lemma toENat_le_iff_of_lt_aleph0 {c c' : Cardinal} (hc' : c' < ℵ₀) :
    toENat c ≤ toENat c' ↔ c ≤ c' := by
  /-
    c c' : Cardinal.{u_1}
    hc' : LT.lt c' Cardinal.aleph0
    ⊢ Iff (LE.le (Cardinal.toENat c) (Cardinal.toENat c')) (LE.le c c')
  -/
  lift c' to ℕ using hc'
  /-
    case intro
    c : Cardinal.{u_1}
    c' : Nat
    ⊢ Iff (LE.le (Cardinal.toENat c) (Cardinal.toENat ↑c')) (LE.le c ↑c')
  -/
  simp_rw [toENat_nat, ← toENat_le_nat]
  /-
    🎉 no goals
  -/


lemma toENat_eq_iff_of_le_aleph0 {c c' : Cardinal} (hc : c ≤ ℵ₀) (hc' : c' ≤ ℵ₀) :
    toENat c = toENat c' ↔ c = c' :=
  toENat_strictMonoOn.injOn.eq_iff hc hc'


@[simp, norm_cast]
                                                           /-
                                                             m n : ENat
                                                             ⊢ Eq (↑(HAdd.hAdd m n)) (HAdd.hAdd ↑m ↑n)
                                                           -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma ofENat_add (m n : ℕ∞) : ofENat (m + n) = m + n := by apply toENat_injOn <;> simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp] lemma aleph0_add_ofENat (m : ℕ∞) : ℵ₀ + m = ℵ₀ := (ofENat_add ⊤ m).symm


                                                             /-
                                                               m : ENat
                                                               ⊢ Eq (HAdd.hAdd (↑m) Cardinal.aleph0) Cardinal.aleph0
                                                             -/
@[simp] lemma ofENat_add_aleph0 (m : ℕ∞) : m + ℵ₀ = ℵ₀ := by rw [add_comm, aleph0_add_ofENat]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp] lemma ofENat_mul_aleph0 {m : ℕ∞} (hm : m ≠ 0) : ↑m * ℵ₀ = ℵ₀ := by
  induction m with
  | top => exact aleph0_mul_aleph0
  | coe m => rw [ofENat_nat, nat_mul_aleph0 (mod_cast hm)]


@[simp] lemma aleph0_mul_ofENat {m : ℕ∞} (hm : m ≠ 0) : ℵ₀ * m = ℵ₀ := by
  /-
    m : ENat
    hm : Ne m 0
    ⊢ Eq (HMul.hMul Cardinal.aleph0 ↑m) Cardinal.aleph0
  -/
  rw [mul_comm, ofENat_mul_aleph0 hm]
  /-
    🎉 no goals
  -/


@[simp] lemma ofENat_mul (m n : ℕ∞) : ofENat (m * n) = m * n :=
                   /-
                     m n : ENat
                     ⊢ Membership.mem (Set.Iic Cardinal.aleph0) ↑(HMul.hMul m n)
                   -/
  toENat_injOn (by simp)
                   /-
                     🎉 no goals
                   -/
                                                                                    /-
                                                                                      m n : ENat
                                                                                      ⊢ Eq (Cardinal.toENat ↑(HMul.hMul m n)) (Cardinal.toENat (HMul.hMul ↑m ↑n))
                                                                                    -/
    (aleph0_mul_aleph0 ▸ mul_le_mul' (ofENat_le_aleph0 _) (ofENat_le_aleph0 _)) (by simp)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- The coercion `Cardinal.ofENat` as a bundled homomorphism. -/
def ofENatHom : ℕ∞ →+*o Cardinal where
  toFun := (↑)
  map_one' := ofENat_one
  map_mul' := ofENat_mul
  map_zero' := ofENat_zero
  map_add' := ofENat_add
  monotone' := ofENat_mono




