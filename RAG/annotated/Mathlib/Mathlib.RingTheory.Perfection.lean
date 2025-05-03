/-- The perfection of a monoid `M`, defined to be the projective limit of `M`
using the `p`-th power maps `M → M` indexed by the natural numbers, implemented as
`{ f : ℕ → M | ∀ n, f (n + 1) ^ p = f n }`. -/
def Monoid.perfection (M : Type u₁) [CommMonoid M] (p : ℕ) : Submonoid (ℕ → M) where
  carrier := { f | ∀ n, f (n + 1) ^ p = f n }
  one_mem' _ := one_pow _
  mul_mem' hf hg n := (mul_pow _ _ _).trans <| congr_arg₂ _ (hf n) (hg n)


/-- The perfection of a ring `R` with characteristic `p`, as a subsemiring,
defined to be the projective limit of `R` using the Frobenius maps `R → R`
indexed by the natural numbers, implemented as `{ f : ℕ → R | ∀ n, f (n + 1) ^ p = f n }`. -/
def Ring.perfectionSubsemiring (R : Type u₁) [CommSemiring R] (p : ℕ) [hp : Fact p.Prime]
    [CharP R p] : Subsemiring (ℕ → R) :=
  { Monoid.perfection R p with
    zero_mem' := fun _ ↦ zero_pow hp.1.ne_zero
    add_mem' := fun hf hg n => (frobenius_add R p _ _).trans <| congr_arg₂ _ (hf n) (hg n) }


/-- The perfection of a ring `R` with characteristic `p`, as a subring,
defined to be the projective limit of `R` using the Frobenius maps `R → R`
indexed by the natural numbers, implemented as `{ f : ℕ → R | ∀ n, f (n + 1) ^ p = f n }`. -/
def Ring.perfectionSubring (R : Type u₁) [CommRing R] (p : ℕ) [hp : Fact p.Prime] [CharP R p] :
    Subring (ℕ → R) :=
  (Ring.perfectionSubsemiring R p).toSubring fun n => by
    /-
      R : Type u₁
      inst✝¹ : CommRing R
      p : Nat
      hp : Fact (Nat.Prime p)
      inst✝ : CharP R p
      n : Nat
      ⊢ Eq (HPow.hPow (Neg.neg 1 (HAdd.hAdd n 1)) p) (Neg.neg 1 n)
    -/
    simp_rw [← frobenius_def, Pi.neg_apply, Pi.one_apply, RingHom.map_neg, RingHom.map_one]
    /-
      🎉 no goals
    -/


/-- The perfection of a ring `R` with characteristic `p`,
defined to be the projective limit of `R` using the Frobenius maps `R → R`
indexed by the natural numbers, implemented as `{f : ℕ → R // ∀ n, f (n + 1) ^ p = f n}`. -/
def Ring.Perfection (R : Type u₁) [CommSemiring R] (p : ℕ) : Type u₁ :=
  { f // ∀ n : ℕ, (f : ℕ → R) (n + 1) ^ p = f n }


instance commSemiring : CommSemiring (Ring.Perfection R p) :=
  (Ring.perfectionSubsemiring R p).toCommSemiring


instance charP : CharP (Ring.Perfection R p) p :=
  CharP.subsemiring (ℕ → R) p (Ring.perfectionSubsemiring R p)


instance ring (R : Type u₁) [CommRing R] [CharP R p] : Ring (Ring.Perfection R p) :=
  (Ring.perfectionSubring R p).toRing


instance commRing (R : Type u₁) [CommRing R] [CharP R p] : CommRing (Ring.Perfection R p) :=
  (Ring.perfectionSubring R p).toCommRing


instance : Inhabited (Ring.Perfection R p) := ⟨0⟩


/-- The `n`-th coefficient of an element of the perfection. -/
def coeff (n : ℕ) : Ring.Perfection R p →+* R where
  toFun f := f.1 n
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl


@[ext]
theorem ext {f g : Ring.Perfection R p} (h : ∀ n, coeff R p n f = coeff R p n g) : f = g :=
  Subtype.eq <| funext h


/-- The `p`-th root of an element of the perfection. -/
def pthRoot : Ring.Perfection R p →+* Ring.Perfection R p where
  toFun f := ⟨fun n => coeff R p (n + 1) f, fun _ => f.2 _⟩
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl


@[simp]
theorem coeff_mk (f : ℕ → R) (hf) (n : ℕ) : coeff R p n ⟨f, hf⟩ = f n := rfl


theorem coeff_pthRoot (f : Ring.Perfection R p) (n : ℕ) :
    coeff R p n (pthRoot R p f) = coeff R p (n + 1) f := rfl


theorem coeff_pow_p (f : Ring.Perfection R p) (n : ℕ) :
                                                    /-
                                                      R : Type u₁
                                                      inst✝¹ : CommSemiring R
                                                      p : Nat
                                                      hp : Fact (Nat.Prime p)
                                                      inst✝ : CharP R p
                                                      f : Ring.Perfection R p
                                                      n : Nat
                                                      ⊢ Eq ((Perfection.coeff R p (HAdd.hAdd n 1)) (HPow.hPow f p)) ((Perfection.coe …
                                                    -/
    coeff R p (n + 1) (f ^ p) = coeff R p n f := by rw [RingHom.map_pow]; exact f.2 n
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem coeff_pow_p' (f : Ring.Perfection R p) (n : ℕ) : coeff R p (n + 1) f ^ p = coeff R p n f :=
  f.2 n


theorem coeff_frobenius (f : Ring.Perfection R p) (n : ℕ) :
                                                              /-
                                                                R : Type u₁
                                                                inst✝¹ : CommSemiring R
                                                                p : Nat
                                                                hp : Fact (Nat.Prime p)
                                                                inst✝ : CharP R p
                                                                f : Ring.Perfection R p
                                                                n : Nat
                                                                ⊢ Eq ((Perfection.coeff R p (HAdd.hAdd n 1)) ((frobenius (Ring.Perfection R p) …
                                                              -/
    coeff R p (n + 1) (frobenius _ p f) = coeff R p n f := by apply coeff_pow_p f n
                                                              /-
                                                                🎉 no goals
                                                              -/

-- `coeff_pow_p f n` also works but is slow!

theorem coeff_iterate_frobenius (f : Ring.Perfection R p) (n m : ℕ) :
    coeff R p (n + m) ((frobenius _ p)^[m] f) = coeff R p n f :=
                                 /-
                                   R : Type u₁
                                   inst✝¹ : CommSemiring R
                                   p : Nat
                                   hp : Fact (Nat.Prime p)
                                   inst✝ : CharP R p
                                   f : Ring.Perfection R p
                                   n m✝ m : Nat
                                   ih : Eq ((Perfection.coeff R p (HAdd.hAdd n m)) (Nat.iterate (⇑(frobenius (Rin …
                                   ⊢ Eq ((Perfection.coeff R p (HAdd.hAdd n m.succ)) (Nat.iterate (⇑(frobenius (R …
                                 -/
  Nat.recOn m rfl fun m ih => by erw [Function.iterate_succ_apply', coeff_frobenius, ih]
                                 /-
                                   🎉 no goals
                                 -/


theorem coeff_iterate_frobenius' (f : Ring.Perfection R p) (n m : ℕ) (hmn : m ≤ n) :
    coeff R p n ((frobenius _ p)^[m] f) = coeff R p (n - m) f :=
  Eq.symm <| (coeff_iterate_frobenius _ _ m).symm.trans <| (tsub_add_cancel_of_le hmn).symm ▸ rfl


theorem pthRoot_frobenius : (pthRoot R p).comp (frobenius _ p) = RingHom.id _ :=
  RingHom.ext fun x =>
                    /-
                      R : Type u₁
                      inst✝¹ : CommSemiring R
                      p : Nat
                      hp : Fact (Nat.Prime p)
                      inst✝ : CharP R p
                      x : Ring.Perfection R p
                      n : Nat
                      ⊢ Eq ((Perfection.coeff R p n) (((Perfection.pthRoot R p).comp (frobenius (Rin …
                    -/
    ext fun n => by rw [RingHom.comp_apply, RingHom.id_apply, coeff_pthRoot, coeff_frobenius]
                    /-
                      🎉 no goals
                    -/


theorem frobenius_pthRoot : (frobenius _ p).comp (pthRoot R p) = RingHom.id _ :=
  RingHom.ext fun x =>
    ext fun n => by
      rw [RingHom.comp_apply, RingHom.id_apply, RingHom.map_frobenius, coeff_pthRoot,
        ← @RingHom.map_frobenius (Ring.Perfection R p) _ R, coeff_frobenius]


theorem coeff_add_ne_zero {f : Ring.Perfection R p} {n : ℕ} (hfn : coeff R p n f ≠ 0) (k : ℕ) :
    coeff R p (n + k) f ≠ 0 :=
  Nat.recOn k hfn fun k ih h => ih <| by
    /-
      R : Type u₁
      inst✝¹ : CommSemiring R
      p : Nat
      hp : Fact (Nat.Prime p)
      inst✝ : CharP R p
      f : Ring.Perfection R p
      n : Nat
      hfn : Ne ((Perfection.coeff R p n) f) 0
      k✝ k : Nat
      ih : Ne ((Perfection.coeff R p (HAdd.hAdd n k)) f) 0
      h : Eq ((Perfection.coeff R p (HAdd.hAdd n k.succ)) f) 0
      ⊢ Eq ((Perfection.coeff R p (HAdd.hAdd n k)) f) 0
    -/
    erw [← coeff_pow_p, RingHom.map_pow, h, zero_pow hp.1.ne_zero]
    /-
      🎉 no goals
    -/


theorem coeff_ne_zero_of_le {f : Ring.Perfection R p} {m n : ℕ} (hfm : coeff R p m f ≠ 0)
    (hmn : m ≤ n) : coeff R p n f ≠ 0 :=
  let ⟨k, hk⟩ := Nat.exists_eq_add_of_le hmn
  hk.symm ▸ coeff_add_ne_zero hfm k


instance perfectRing : PerfectRing (Ring.Perfection R p) p where
  bijective_frobenius := Function.bijective_iff_has_inverse.mpr
    ⟨pthRoot R p,
     DFunLike.congr_fun <| @frobenius_pthRoot R _ p _ _,
     DFunLike.congr_fun <| @pthRoot_frobenius R _ p _ _⟩


/-- Given rings `R` and `S` of characteristic `p`, with `R` being perfect,
any homomorphism `R →+* S` can be lifted to a homomorphism `R →+* Perfection S p`. -/
@[simps]
noncomputable def lift (R : Type u₁) [CommSemiring R] [CharP R p] [PerfectRing R p]
    (S : Type u₂) [CommSemiring S] [CharP S p] : (R →+* S) ≃ (R →+* Ring.Perfection S p) where
  toFun f :=
    { toFun := fun r => ⟨fun n => f (((frobeniusEquiv R p).symm : R →+* R)^[n] r),
                    /-
                      R✝ : Type u₁
                      inst✝⁶ : CommSemiring R✝
                      p : Nat
                      hp : Fact (Nat.Prime p)
                      inst✝⁵ : CharP R✝ p
                      R : Type u₁
                      inst✝⁴ : CommSemiring R
                      inst✝³ : CharP R p
                      inst✝² : PerfectRing R p
                      S : Type u₂
                      inst✝¹ : CommSemiring S
                      inst✝ : CharP S p
                      f : RingHom R S
                      r : R
                      n : Nat
                      ⊢ Eq (HPow.hPow ((fun n => f (Nat.iterate (⇑↑(frobeniusEquiv R p).symm) n r))  …
                    -/
        fun n => by erw [← f.map_pow, Function.iterate_succ_apply', frobeniusEquiv_symm_pow_p]⟩
                    /-
                      🎉 no goals
                    -/
      map_one' := ext fun _ => (congr_arg f <| iterate_map_one _ _).trans f.map_one
      map_mul' := fun _ _ =>
        ext fun _ => (congr_arg f <| iterate_map_mul _ _ _ _).trans <| f.map_mul _ _
      map_zero' := ext fun _ => (congr_arg f <| iterate_map_zero _ _).trans f.map_zero
      map_add' := fun _ _ =>
        ext fun _ => (congr_arg f <| iterate_map_add _ _ _ _).trans <| f.map_add _ _ }
  invFun := RingHom.comp <| coeff S p 0
  left_inv _ := RingHom.ext fun _ => rfl
  right_inv f := RingHom.ext fun r => ext fun n =>
    show coeff S p 0 (f (((frobeniusEquiv R p).symm)^[n] r)) = coeff S p n (f r) by
      rw [← coeff_iterate_frobenius _ 0 n, zero_add, ← RingHom.map_iterate_frobenius,
        Function.RightInverse.iterate (frobenius_apply_frobeniusEquiv_symm R p) n]


theorem hom_ext {R : Type u₁} [CommSemiring R] [CharP R p] [PerfectRing R p] {S : Type u₂}
    [CommSemiring S] [CharP S p] {f g : R →+* Ring.Perfection S p}
    (hfg : ∀ x, coeff S p 0 (f x) = coeff S p 0 (g x)) : f = g :=
  (lift p R S).symm.injective <| RingHom.ext hfg


/-- A ring homomorphism `R →+* S` induces `Perfection R p →+* Perfection S p`. -/
@[simps]
def map (φ : R →+* S) : Ring.Perfection R p →+* Ring.Perfection S p where
                                                      /-
                                                        R : Type u₁
                                                        inst✝³ : CommSemiring R
                                                        p : Nat
                                                        hp : Fact (Nat.Prime p)
                                                        inst✝² : CharP R p
                                                        S : Type u₂
                                                        inst✝¹ : CommSemiring S
                                                        inst✝ : CharP S p
                                                        φ : RingHom R S
                                                        f : Ring.Perfection R p
                                                        n : Nat
                                                        ⊢ Eq (HPow.hPow ((fun n => φ ((Perfection.coeff R p n) f)) (HAdd.hAdd n 1)) p) …
                                                      -/
  toFun f := ⟨fun n => φ (coeff R p n f), fun n => by rw [← φ.map_pow, coeff_pow_p']⟩
                                                      /-
                                                        🎉 no goals
                                                      -/
  map_one' := Subtype.eq <| funext fun _ => φ.map_one
  map_mul' _ _ := Subtype.eq <| funext fun _ => φ.map_mul _ _
  map_zero' := Subtype.eq <| funext fun _ => φ.map_zero
  map_add' _ _ := Subtype.eq <| funext fun _ => φ.map_add _ _


theorem coeff_map (φ : R →+* S) (f : Ring.Perfection R p) (n : ℕ) :
    coeff S p n (map p φ f) = φ (coeff R p n f) := rfl


/-- A perfection map to a ring of characteristic `p` is a map that is isomorphic
to its perfection. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
structure PerfectionMap (p : ℕ) [Fact p.Prime] {R : Type u₁} [CommSemiring R] [CharP R p]
    {P : Type u₂} [CommSemiring P] [CharP P p] [PerfectRing P p] (π : P →+* R) : Prop where
  injective : ∀ ⦃x y : P⦄,
    (∀ n, π (((frobeniusEquiv P p).symm)^[n] x) = π (((frobeniusEquiv P p).symm)^[n] y)) → x = y
  surjective : ∀ f : ℕ → R, (∀ n, f (n + 1) ^ p = f n) → ∃ x : P, ∀ n,
    π (((frobeniusEquiv P p).symm)^[n] x) = f n


/-- Create a `PerfectionMap` from an isomorphism to the perfection. -/
@[simps]
theorem mk' {f : P →+* R} (g : P ≃+* Ring.Perfection R p) (hfg : Perfection.lift p P R f = g) :
    PerfectionMap p f :=
  { injective := fun x y hxy =>
      g.injective <|
        (RingHom.ext_iff.1 hfg x).symm.trans <|
          Eq.symm <| (RingHom.ext_iff.1 hfg y).symm.trans <| Perfection.ext fun n => (hxy n).symm
    surjective := fun y hy =>
      let ⟨x, hx⟩ := g.surjective ⟨y, hy⟩
      ⟨x, fun n =>
        show Perfection.coeff R p n (Perfection.lift p P R f x) = Perfection.coeff R p n ⟨y, hy⟩ by
          /-
            p : Nat
            inst✝⁵ : Fact (Nat.Prime p)
            R : Type u₁
            inst✝⁴ : CommSemiring R
            inst✝³ : CharP R p
            P : Type u₃
            inst✝² : CommSemiring P
            inst✝¹ : CharP P p
            inst✝ : PerfectRing P p
            f : RingHom P R
            g : RingEquiv P (Ring.Perfection R p)
            hfg : Eq ((Perfection.lift p P R) f) ↑g
            y : Nat → R
            hy : ∀ (n : Nat), Eq (HPow.hPow (y (HAdd.hAdd n 1)) p) (y n)
            x : P
            hx : Eq (g x) ⟨y, hy⟩
            n : Nat
            ⊢ Eq ((Perfection.coeff R p n) (((Perfection.lift p P R) f) x)) ((Perfection.c …
          -/
          simp [hfg, hx]⟩ }
          /-
            🎉 no goals
          -/


/-- The canonical perfection map from the perfection of a ring. -/
theorem of : PerfectionMap p (Perfection.coeff R p 0) :=
  mk' (RingEquiv.refl _) <| (Equiv.apply_eq_iff_eq_symm_apply _).2 rfl


/-- For a perfect ring, it itself is the perfection. -/
theorem id [PerfectRing R p] : PerfectionMap p (RingHom.id R) :=
  { injective := fun _ _ hxy => hxy 0
    surjective := fun f hf =>
      ⟨f 0, fun n =>
        show ((frobeniusEquiv R p).symm)^[n] (f 0) = f n from
          Nat.recOn n rfl fun n ih => injective_pow_p R p <| by
            /-
              p : Nat
              inst✝³ : Fact (Nat.Prime p)
              R : Type u₁
              inst✝² : CommSemiring R
              inst✝¹ : CharP R p
              inst✝ : PerfectRing R p
              f : Nat → R
              hf : ∀ (n : Nat), Eq (HPow.hPow (f (HAdd.hAdd n 1)) p) (f n)
              n✝ n : Nat
              ih : Eq (Nat.iterate (⇑(frobeniusEquiv R p).symm) n (f 0)) (f n)
              ⊢ Eq (HPow.hPow (Nat.iterate (⇑(frobeniusEquiv R p).symm) n.succ (f 0)) p) (HP …
            -/
            rw [Function.iterate_succ_apply', frobeniusEquiv_symm_pow_p, ih, hf]⟩ }
            /-
              🎉 no goals
            -/


/-- A perfection map induces an isomorphism to the perfection. -/
noncomputable def equiv {π : P →+* R} (m : PerfectionMap p π) : P ≃+* Ring.Perfection R p :=
  RingEquiv.ofBijective (Perfection.lift p P R π)
    ⟨fun _ _ hxy => m.injective fun n => (congr_arg (Perfection.coeff R p n) hxy : _), fun f =>
      let ⟨x, hx⟩ := m.surjective f.1 f.2
      ⟨x, Perfection.ext <| hx⟩⟩


theorem equiv_apply {π : P →+* R} (m : PerfectionMap p π) (x : P) :
    m.equiv x = Perfection.lift p P R π x := rfl


theorem comp_equiv {π : P →+* R} (m : PerfectionMap p π) (x : P) :
    Perfection.coeff R p 0 (m.equiv x) = π x := rfl


theorem comp_equiv' {π : P →+* R} (m : PerfectionMap p π) :
    (Perfection.coeff R p 0).comp ↑m.equiv = π :=
  RingHom.ext fun _ => rfl


theorem comp_symm_equiv {π : P →+* R} (m : PerfectionMap p π) (f : Ring.Perfection R p) :
    π (m.equiv.symm f) = Perfection.coeff R p 0 f :=
  (m.comp_equiv _).symm.trans <| congr_arg _ <| m.equiv.apply_symm_apply f


theorem comp_symm_equiv' {π : P →+* R} (m : PerfectionMap p π) :
    π.comp ↑m.equiv.symm = Perfection.coeff R p 0 :=
  RingHom.ext m.comp_symm_equiv


/-- Given rings `R` and `S` of characteristic `p`, with `R` being perfect,
any homomorphism `R →+* S` can be lifted to a homomorphism `R →+* P`,
where `P` is any perfection of `S`. -/
@[simps]
noncomputable def lift [PerfectRing R p] (S : Type u₂) [CommSemiring S] [CharP S p] (P : Type u₃)
    [CommSemiring P] [CharP P p] [PerfectRing P p] (π : P →+* S) (m : PerfectionMap p π) :
    (R →+* S) ≃ (R →+* P) where
  toFun f := RingHom.comp ↑m.equiv.symm <| Perfection.lift p R S f
  invFun f := π.comp f
  left_inv f := by
    /-
      p : Nat
      inst✝¹¹ : Fact (Nat.Prime p)
      R : Type u₁
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CharP R p
      P✝ : Type u₃
      inst✝⁸ : CommSemiring P✝
      inst✝⁷ : CharP P✝ p
      inst✝⁶ : PerfectRing P✝ p
      inst✝⁵ : PerfectRing R p
      S : Type u₂
      inst✝⁴ : CommSemiring S
      inst✝³ : CharP S p
      P : Type u₃
      inst✝² : CommSemiring P
      inst✝¹ : CharP P p
      inst✝ : PerfectRing P p
      π : RingHom P S
      m : PerfectionMap p π
      f : RingHom R S
      ⊢ Eq ((fun f => π.comp f) ((fun f => (↑m.equiv.symm).comp ((Perfection.lift p  …
    -/
    simp_rw [← RingHom.comp_assoc, comp_symm_equiv']
    /-
      p : Nat
      inst✝¹¹ : Fact (Nat.Prime p)
      R : Type u₁
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CharP R p
      P✝ : Type u₃
      inst✝⁸ : CommSemiring P✝
      inst✝⁷ : CharP P✝ p
      inst✝⁶ : PerfectRing P✝ p
      inst✝⁵ : PerfectRing R p
      S : Type u₂
      inst✝⁴ : CommSemiring S
      inst✝³ : CharP S p
      P : Type u₃
      inst✝² : CommSemiring P
      inst✝¹ : CharP P p
      inst✝ : PerfectRing P p
      π : RingHom P S
      m : PerfectionMap p π
      f : RingHom R S
      ⊢ Eq ((Perfection.coeff S p 0).comp ((Perfection.lift p R S) f)) f
    -/
    exact (Perfection.lift p R S).symm_apply_apply f
    /-
      🎉 no goals
    -/
  right_inv f := by
    exact RingHom.ext fun x => m.equiv.injective <| (m.equiv.apply_symm_apply _).trans
      <| show Perfection.lift p R S (π.comp f) x = RingHom.comp (↑m.equiv) f x from
        RingHom.ext_iff.1 (by rw [Equiv.apply_eq_iff_eq_symm_apply]; rfl) _


theorem hom_ext [PerfectRing R p] {S : Type u₂} [CommSemiring S] [CharP S p] {P : Type u₃}
    [CommSemiring P] [CharP P p] [PerfectRing P p] (π : P →+* S) (m : PerfectionMap p π)
    {f g : R →+* P} (hfg : ∀ x, π (f x) = π (g x)) : f = g :=
  (lift p R S P π m).symm.injective <| RingHom.ext hfg


/-- A ring homomorphism `R →+* S` induces `P →+* Q`, a map of the respective perfections. -/
@[nolint unusedArguments]
noncomputable def map {π : P →+* R} (_ : PerfectionMap p π) {σ : Q →+* S} (n : PerfectionMap p σ)
    (φ : R →+* S) : P →+* Q :=
  lift p P S Q σ n <| φ.comp π


theorem comp_map {π : P →+* R} (m : PerfectionMap p π) {σ : Q →+* S} (n : PerfectionMap p σ)
    (φ : R →+* S) : σ.comp (map p m n φ) = φ.comp π :=
  (lift p P S Q σ n).symm_apply_apply _


theorem map_map {π : P →+* R} (m : PerfectionMap p π) {σ : Q →+* S} (n : PerfectionMap p σ)
    (φ : R →+* S) (x : P) : σ (map p m n φ x) = φ (π x) :=
  RingHom.ext_iff.1 (comp_map p m n φ) x


theorem map_eq_map (φ : R →+* S) : map p (of p R) (of p S) φ = Perfection.map p φ :=
                                 /-
                                   p : Nat
                                   inst✝⁴ : Fact (Nat.Prime p)
                                   R : Type u₁
                                   inst✝³ : CommSemiring R
                                   inst✝² : CharP R p
                                   S : Type u₂
                                   inst✝¹ : CommSemiring S
                                   inst✝ : CharP S p
                                   φ : RingHom R S
                                   f : Ring.Perfection R p
                                   ⊢ Eq ((Perfection.coeff S p 0) ((PerfectionMap.map p ⋯ ⋯ φ) f)) ((Perfection.c …
                                 -/
  hom_ext _ (of p S) fun f => by rw [map_map, Perfection.coeff_map]
                                 /-
                                   🎉 no goals
                                 -/


/-- `O/(p)` for `O`, ring of integers of `K`. -/
@[nolint unusedArguments] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `nolint has_nonempty_instance`
def ModP (K : Type u₁) [Field K] (v : Valuation K ℝ≥0) (O : Type u₂) [CommRing O] [Algebra O K]
    (_ : v.Integers O) (p : ℕ) :=
  O ⧸ (Ideal.span {(p : O)} : Ideal O)


instance commRing : CommRing (ModP K v O hv p) :=
  Ideal.Quotient.commRing (Ideal.span {(p : O)} : Ideal O)


instance charP [Fact p.Prime] [hvp : Fact (v p ≠ 1)] : CharP (ModP K v O hv p) p :=
  CharP.quotient O p <| mt hv.one_of_isUnit <| (map_natCast (algebraMap O K) p).symm ▸ hvp.1


instance [hp : Fact p.Prime] [Fact (v p ≠ 1)] : Nontrivial (ModP K v O hv p) :=
  CharP.nontrivial_of_char_ne_one hp.1.ne_one


/-- For a field `K` with valuation `v : K → ℝ≥0` and ring of integers `O`,
a function `O/(p) → ℝ≥0` that sends `0` to `0` and `x + (p)` to `v(x)` as long as `x ∉ (p)`. -/
noncomputable def preVal (x : ModP K v O hv p) : ℝ≥0 :=
  if x = 0 then 0 else v (algebraMap O K x.out)


theorem preVal_mk {x : O} (hx : (Ideal.Quotient.mk _ x : ModP K v O hv p) ≠ 0) :
    preVal K v O hv p (Ideal.Quotient.mk _ x) = v (algebraMap O K x) := by
  obtain ⟨r, hr⟩ : ∃ (a : O), a * (p : O) = (Quotient.mk'' x).out - x :=
    Ideal.mem_span_singleton'.1 <| Ideal.Quotient.eq.1 <| Quotient.sound' <| Quotient.mk_out' _
  /-
    case intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x : O
    hx : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) x) 0
    r : O
    hr : Eq (HMul.hMul r ↑p) (HSub.hSub (Quotient.mk'' x).out x)
    ⊢ Eq (ModP.preVal K v O hv p ((Ideal.Quotient.mk (Ideal.span (Singleton.single …
  -/
  refine (if_neg hx).trans (v.map_eq_of_sub_lt <| lt_of_not_le ?_)
  /-
    case intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x : O
    hx : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) x) 0
    r : O
    hr : Eq (HMul.hMul r ↑p) (HSub.hSub (Quotient.mk'' x).out x)
    ⊢ Not (LE.le (v ((algebraMap O K) x)) (v (HSub.hSub ((algebraMap O K) (Quotien …
  -/
  erw [← RingHom.map_sub, ← hr, hv.le_iff_dvd]
  exact fun hprx =>
    hx (Ideal.Quotient.eq_zero_iff_mem.2 <| Ideal.mem_span_singleton.2 <| dvd_of_mul_left_dvd hprx)


theorem preVal_zero : preVal K v O hv p 0 = 0 :=
  if_pos rfl


theorem preVal_mul {x y : ModP K v O hv p} (hxy0 : x * y ≠ 0) :
    preVal K v O hv p (x * y) = preVal K v O hv p x * preVal K v O hv p y := by
  /-
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    hxy0 : Ne (HMul.hMul x y) 0
    ⊢ Eq (ModP.preVal K v O hv p (HMul.hMul x y)) (HMul.hMul (ModP.preVal K v O hv …
  -/
  have hx0 : x ≠ 0 := mt (by rintro rfl; rw [zero_mul]) hxy0
  /-
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    hxy0 : Ne (HMul.hMul x y) 0
    hx0 : Ne x 0
    ⊢ Eq (ModP.preVal K v O hv p (HMul.hMul x y)) (HMul.hMul (ModP.preVal K v O hv …
  -/
  have hy0 : y ≠ 0 := mt (by rintro rfl; rw [mul_zero]) hxy0
  /-
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    hxy0 : Ne (HMul.hMul x y) 0
    hx0 : Ne x 0
    hy0 : Ne y 0
    ⊢ Eq (ModP.preVal K v O hv p (HMul.hMul x y)) (HMul.hMul (ModP.preVal K v O hv …
  -/
  obtain ⟨r, rfl⟩ := Ideal.Quotient.mk_surjective x
  /-
    case intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    y : ModP K v O hv p
    hy0 : Ne y 0
    r : O
    hxy0 : Ne (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)) …
    hx0 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
    ⊢ Eq (ModP.preVal K v O hv p (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singl …
  -/
  obtain ⟨s, rfl⟩ := Ideal.Quotient.mk_surjective y
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    r : O
    hx0 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
    s : O
    hy0 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) s) 0
    hxy0 : Ne (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)) …
    ⊢ Eq (ModP.preVal K v O hv p (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singl …
  -/
  rw [← map_mul (Ideal.Quotient.mk (Ideal.span {↑p})) r s] at hxy0 ⊢
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    r : O
    hx0 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
    s : O
    hy0 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) s) 0
    hxy0 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HMul.hMu …
    ⊢ Eq (ModP.preVal K v O hv p ((Ideal.Quotient.mk (Ideal.span (Singleton.single …
  -/
  rw [preVal_mk hx0, preVal_mk hy0, preVal_mk hxy0, RingHom.map_mul, v.map_mul]
  /-
    🎉 no goals
  -/


theorem preVal_add (x y : ModP K v O hv p) :
    preVal K v O hv p (x + y) ≤ max (preVal K v O hv p x) (preVal K v O hv p y) := by
  /-
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      K : Type u₁
      inst✝² : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝¹ : CommRing O
      inst✝ : Algebra O K
      hv : v.Integers O
      p : Nat
      x y : ModP K v O hv p
      hx0 : Eq x 0
      ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
    -/
  · rw [hx0, zero_add]; exact le_max_right _ _
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    hx0 : Not (Eq x 0)
    ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
  -/
  by_cases hy0 : y = 0
    /-
      case pos
      K : Type u₁
      inst✝² : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝¹ : CommRing O
      inst✝ : Algebra O K
      hv : v.Integers O
      p : Nat
      x y : ModP K v O hv p
      hx0 : Not (Eq x 0)
      hy0 : Eq y 0
      ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
    -/
  · rw [hy0, add_zero]; exact le_max_left _ _
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    hx0 : Not (Eq x 0)
    hy0 : Not (Eq y 0)
    ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
  -/
  by_cases hxy0 : x + y = 0
    /-
      case pos
      K : Type u₁
      inst✝² : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝¹ : CommRing O
      inst✝ : Algebra O K
      hv : v.Integers O
      p : Nat
      x y : ModP K v O hv p
      hx0 : Not (Eq x 0)
      hy0 : Not (Eq y 0)
      hxy0 : Eq (HAdd.hAdd x y) 0
      ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
    -/
  · rw [hxy0, preVal_zero]; exact zero_le _
                            /-
                              🎉 no goals
                            -/
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x y : ModP K v O hv p
    hx0 : Not (Eq x 0)
    hy0 : Not (Eq y 0)
    hxy0 : Not (Eq (HAdd.hAdd x y) 0)
    ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd x y)) (Max.max (ModP.preVal K v O h …
  -/
  obtain ⟨r, rfl⟩ := Ideal.Quotient.mk_surjective x
  /-
    case neg.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    y : ModP K v O hv p
    hy0 : Not (Eq y 0)
    r : O
    hx0 : Not (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0)
    hxy0 : Not (Eq (HAdd.hAdd ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton …
    ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd ((Ideal.Quotient.mk (Ideal.span (Si …
  -/
  obtain ⟨s, rfl⟩ := Ideal.Quotient.mk_surjective y
  /-
    case neg.intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    r : O
    hx0 : Not (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0)
    s : O
    hy0 : Not (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) s) 0)
    hxy0 : Not (Eq (HAdd.hAdd ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton …
    ⊢ LE.le (ModP.preVal K v O hv p (HAdd.hAdd ((Ideal.Quotient.mk (Ideal.span (Si …
  -/
  rw [← map_add (Ideal.Quotient.mk (Ideal.span {↑p})) r s] at hxy0 ⊢
  /-
    case neg.intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    r : O
    hx0 : Not (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0)
    s : O
    hy0 : Not (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) s) 0)
    hxy0 : Not (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HAd …
    ⊢ LE.le (ModP.preVal K v O hv p ((Ideal.Quotient.mk (Ideal.span (Singleton.sin …
  -/
  rw [preVal_mk hx0, preVal_mk hy0, preVal_mk hxy0, RingHom.map_add]; exact v.map_add _ _
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem v_p_lt_preVal {x : ModP K v O hv p} : v p < preVal K v O hv p x ↔ x ≠ 0 := by
  refine ⟨fun h hx => by rw [hx, preVal_zero] at h; exact not_lt_zero' h,
    fun h => lt_of_not_le fun hp => h ?_⟩
  /-
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    x : ModP K v O hv p
    h : Ne x 0
    hp : LE.le (ModP.preVal K v O hv p x) (v ↑p)
    ⊢ Eq x 0
  -/
  obtain ⟨r, rfl⟩ := Ideal.Quotient.mk_surjective x
  /-
    case intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    r : O
    h : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
    hp : LE.le (ModP.preVal K v O hv p ((Ideal.Quotient.mk (Ideal.span (Singleton. …
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
  -/
  rw [preVal_mk h, ← map_natCast (algebraMap O K) p, hv.le_iff_dvd] at hp
  /-
    case intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    r : O
    h : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
    hp : Dvd.dvd (↑p) r
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r) 0
  -/
  rw [Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_span_singleton]; exact hp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem preVal_eq_zero {x : ModP K v O hv p} : preVal K v O hv p x = 0 ↔ x = 0 :=
  ⟨fun hvx =>
    by_contradiction fun hx0 : x ≠ 0 => by
      /-
        K : Type u₁
        inst✝² : Field K
        v : Valuation K NNReal
        O : Type u₂
        inst✝¹ : CommRing O
        inst✝ : Algebra O K
        hv : v.Integers O
        p : Nat
        x : ModP K v O hv p
        hvx : Eq (ModP.preVal K v O hv p x) 0
        hx0 : Ne x 0
        ⊢ False
      -/
      rw [← v_p_lt_preVal, hvx] at hx0
      /-
        K : Type u₁
        inst✝² : Field K
        v : Valuation K NNReal
        O : Type u₂
        inst✝¹ : CommRing O
        inst✝ : Algebra O K
        hv : v.Integers O
        p : Nat
        x : ModP K v O hv p
        hvx : Eq (ModP.preVal K v O hv p x) 0
        hx0 : LT.lt (v ↑p) 0
        ⊢ False
      -/
      exact not_lt_zero' hx0,
      /-
        🎉 no goals
      -/
    fun hx => hx.symm ▸ preVal_zero⟩


theorem v_p_lt_val {x : O} :
    v p < v (algebraMap O K x) ↔ (Ideal.Quotient.mk _ x : ModP K v O hv p) ≠ 0 := by
  rw [lt_iff_not_le, not_iff_not, ← map_natCast (algebraMap O K) p, hv.le_iff_dvd,
    Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_span_singleton]


theorem mul_ne_zero_of_pow_p_ne_zero {x y : ModP K v O hv p} (hx : x ^ p ≠ 0) (hy : y ^ p ≠ 0) :
    x * y ≠ 0 := by
  /-
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : ModP K v O hv p
    hx : Ne (HPow.hPow x p) 0
    hy : Ne (HPow.hPow y p) 0
    ⊢ Ne (HMul.hMul x y) 0
  -/
  obtain ⟨r, rfl⟩ := Ideal.Quotient.mk_surjective x
  /-
    case intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    y : ModP K v O hv p
    hy : Ne (HPow.hPow y p) 0
    r : O
    hx : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    ⊢ Ne (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r)  …
  -/
  obtain ⟨s, rfl⟩ := Ideal.Quotient.mk_surjective y
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    s : O
    hy : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    ⊢ Ne (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r)  …
  -/
  have h1p : (0 : ℝ) < 1 / p := one_div_pos.2 (Nat.cast_pos.2 hp.1.pos)
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    s : O
    hy : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    ⊢ Ne (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) r)  …
  -/
  rw [← (Ideal.Quotient.mk (Ideal.span {(p : O)})).map_mul]
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    s : O
    hy : Ne (HPow.hPow ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p)))  …
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    ⊢ Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HMul.hMul r s …
  -/
  rw [← (Ideal.Quotient.mk (Ideal.span {(p : O)})).map_pow] at hx hy
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HPow.hPow  …
    s : O
    hy : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HPow.hPow  …
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    ⊢ Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HMul.hMul r s …
  -/
  rw [← v_p_lt_val hv] at hx hy ⊢
  rw [RingHom.map_pow, v.map_pow, ← rpow_lt_rpow_iff h1p, ← rpow_natCast, ← rpow_mul,
    mul_one_div_cancel (Nat.cast_ne_zero.2 hp.1.ne_zero : (p : ℝ) ≠ 0), rpow_one] at hx hy
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    ⊢ LT.lt (v ↑p) (v ((algebraMap O K) (HMul.hMul r s)))
  -/
  rw [RingHom.map_mul, v.map_mul]; refine lt_of_le_of_lt ?_ (mul_lt_mul'' hx hy zero_le' zero_le')
  /-
    case intro.intro
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    ⊢ LE.le (v ↑p) (HMul.hMul (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (HPow.hPow (v ↑p …
  -/
  by_cases hvp : v p = 0
    /-
      case pos
      K : Type u₁
      inst✝² : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝¹ : CommRing O
      inst✝ : Algebra O K
      hv : v.Integers O
      p : Nat
      hp : Fact (Nat.Prime p)
      r : O
      hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
      s : O
      hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
      h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
      hvp : Eq (v ↑p) 0
      ⊢ LE.le (v ↑p) (HMul.hMul (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (HPow.hPow (v ↑p …
    -/
  · rw [hvp]; exact zero_le _
              /-
                🎉 no goals
              -/
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    hvp : Not (Eq (v ↑p) 0)
    ⊢ LE.le (v ↑p) (HMul.hMul (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (HPow.hPow (v ↑p …
  -/
  replace hvp := zero_lt_iff.2 hvp
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    hvp : LT.lt 0 (v ↑p)
    ⊢ LE.le (v ↑p) (HMul.hMul (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (HPow.hPow (v ↑p …
  -/
  conv_lhs => rw [← rpow_one (v p)]
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    hvp : LT.lt 0 (v ↑p)
    ⊢ LE.le (HPow.hPow (v ↑p) 1) (HMul.hMul (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (H …
  -/
  rw [← rpow_add (ne_of_gt hvp)]
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    hvp : LT.lt 0 (v ↑p)
    ⊢ LE.le (HPow.hPow (v ↑p) 1) (HPow.hPow (v ↑p) (HAdd.hAdd (HDiv.hDiv 1 ↑p) (HD …
  -/
  refine rpow_le_rpow_of_exponent_ge hvp (map_natCast (algebraMap O K) p ▸ hv.2 _) ?_
  /-
    case neg
    K : Type u₁
    inst✝² : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝¹ : CommRing O
    inst✝ : Algebra O K
    hv : v.Integers O
    p : Nat
    hp : Fact (Nat.Prime p)
    r : O
    hx : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) r))
    s : O
    hy : LT.lt (HPow.hPow (v ↑p) (HDiv.hDiv 1 ↑p)) (v ((algebraMap O K) s))
    h1p : LT.lt 0 (HDiv.hDiv 1 ↑p)
    hvp : LT.lt 0 (v ↑p)
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv 1 ↑p) (HDiv.hDiv 1 ↑p)) 1
  -/
  rw [← add_div, div_le_one (Nat.cast_pos.2 hp.1.pos : 0 < (p : ℝ))]; exact mod_cast hp.1.two_le
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- Perfection of `O/(p)` where `O` is the ring of integers of `K`. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
def PreTilt :=
  Ring.Perfection (ModP K v O hv p) p


instance : CommRing (PreTilt K v O hv p) :=
  Perfection.commRing p _


instance : CharP (PreTilt K v O hv p) p :=
  Perfection.charP (ModP K v O hv p) p


/-- The valuation `Perfection(O/(p)) → ℝ≥0` as a function.
Given `f ∈ Perfection(O/(p))`, if `f = 0` then output `0`;
otherwise output `preVal(f(n))^(p^n)` for any `n` such that `f(n) ≠ 0`. -/
noncomputable def valAux (f : PreTilt K v O hv p) : ℝ≥0 :=
  if h : ∃ n, coeff _ _ n f ≠ 0 then
    ModP.preVal K v O hv p (coeff _ _ (Nat.find h) f) ^ p ^ Nat.find h
  else 0


theorem coeff_nat_find_add_ne_zero {f : PreTilt K v O hv p} {h : ∃ n, coeff _ _ n f ≠ 0} (k : ℕ) :
    coeff _ _ (Nat.find h + k) f ≠ 0 :=
  coeff_add_ne_zero (Nat.find_spec h) k


theorem valAux_eq {f : PreTilt K v O hv p} {n : ℕ} (hfn : coeff _ _ n f ≠ 0) :
    valAux K v O hv p f = ModP.preVal K v O hv p (coeff _ _ n f) ^ p ^ n := by
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    n : Nat
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    ⊢ Eq (PreTilt.valAux K v O hv p f) (HPow.hPow (ModP.preVal K v O hv p ((Perfec …
  -/
  have h : ∃ n, coeff _ _ n f ≠ 0 := ⟨n, hfn⟩
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    n : Nat
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    ⊢ Eq (PreTilt.valAux K v O hv p f) (HPow.hPow (ModP.preVal K v O hv p ((Perfec …
  -/
  rw [valAux, dif_pos h]
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    n : Nat
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    ⊢ Eq (HPow.hPow (ModP.preVal K v O hv p ((Perfection.coeff (ModP K v O hv p) p …
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le (Nat.find_min' h hfn)
  /-
    case intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    k : Nat
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) k)) f) 0
    ⊢ Eq (HPow.hPow (ModP.preVal K v O hv p ((Perfection.coeff (ModP K v O hv p) p …
  -/
  induction' k with k ih
    /-
      case intro.zero
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f : PreTilt K v O hv p
      h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
      hfn : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) 0)) f) 0
      ⊢ Eq (HPow.hPow (ModP.preVal K v O hv p ((Perfection.coeff (ModP K v O hv p) p …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    k : Nat
    ih : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) k)) f)  …
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) (HAdd. …
    ⊢ Eq (HPow.hPow (ModP.preVal K v O hv p ((Perfection.coeff (ModP K v O hv p) p …
  -/
  obtain ⟨x, hx⟩ := Ideal.Quotient.mk_surjective (coeff (ModP K v O hv p) p (Nat.find h + k + 1) f)
  /-
    case intro.succ.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    k : Nat
    ih : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) k)) f)  …
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) (HAdd. …
    x : O
    hx : Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) x) ((Perfec …
    ⊢ Eq (HPow.hPow (ModP.preVal K v O hv p ((Perfection.coeff (ModP K v O hv p) p …
  -/
  have h1 : (Ideal.Quotient.mk _ x : ModP K v O hv p) ≠ 0 := hx.symm ▸ hfn
  have h2 : (Ideal.Quotient.mk _ (x ^ p) : ModP K v O hv p) ≠ 0 := by
    erw [RingHom.map_pow, hx, ← RingHom.map_pow, coeff_pow_p]
    exact coeff_nat_find_add_ne_zero k
  erw [ih (coeff_nat_find_add_ne_zero k), ← hx, ← coeff_pow_p, RingHom.map_pow, ← hx,
    ← RingHom.map_pow, ModP.preVal_mk h1, ModP.preVal_mk h2, RingHom.map_pow, v.map_pow, ← pow_mul,
    pow_succ']
  /-
    case intro.succ.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    h : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    k : Nat
    ih : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) k)) f)  …
    hfn : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Nat.find h) (HAdd. …
    x : O
    hx : Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) x) ((Perfec …
    h1 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) x) 0
    h2 : Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) (HPow.hPow  …
    ⊢ Eq (HPow.hPow (v ((algebraMap O K) x)) (HMul.hMul p (HPow.hPow p (HAdd.hAdd  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem valAux_zero : valAux K v O hv p 0 = 0 :=
  dif_neg fun ⟨_, hn⟩ => hn rfl


theorem valAux_one : valAux K v O hv p 1 = 1 :=
  (valAux_eq <| show coeff (ModP K v O hv p) p 0 1 ≠ 0 from one_ne_zero).trans <| by
    rw [pow_zero, pow_one, RingHom.map_one, ← (Ideal.Quotient.mk _).map_one, ModP.preVal_mk,
      RingHom.map_one, v.map_one]
    /-
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      ⊢ Ne ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑p))) 1) 0
    -/
    change (1 : ModP K v O hv p) ≠ 0
    /-
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      ⊢ Ne 1 0
    -/
    exact one_ne_zero
    /-
      🎉 no goals
    -/


theorem valAux_mul (f g : PreTilt K v O hv p) :
    valAux K v O hv p (f * g) = valAux K v O hv p f * valAux K v O hv p g := by
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  by_cases hf : f = 0
    /-
      case pos
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      hf : Eq f 0
      ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
    -/
  · rw [hf, zero_mul, valAux_zero, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  by_cases hg : g = 0
    /-
      case pos
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      hf : Not (Eq f 0)
      hg : Eq g 0
      ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
    -/
  · rw [hg, mul_zero, valAux_zero, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  obtain ⟨m, hm⟩ : ∃ n, coeff _ _ n f ≠ 0 := not_forall.1 fun h => hf <| Perfection.ext h
  /-
    case neg.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    m : Nat
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p m) f) 0
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  obtain ⟨n, hn⟩ : ∃ n, coeff _ _ n g ≠ 0 := not_forall.1 fun h => hg <| Perfection.ext h
  /-
    case neg.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    m : Nat
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p m) f) 0
    n : Nat
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p n) g) 0
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  replace hm := coeff_ne_zero_of_le hm (le_max_left m n)
  /-
    case neg.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    m n : Nat
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p n) g) 0
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max m n)) f) 0
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  replace hn := coeff_ne_zero_of_le hn (le_max_right m n)
  have hfg : coeff _ _ (max m n + 1) (f * g) ≠ 0 := by
    rw [RingHom.map_mul]
    refine ModP.mul_ne_zero_of_pow_p_ne_zero ?_ ?_
    · rw [← RingHom.map_pow, coeff_pow_p f]; assumption
    · rw [← RingHom.map_pow, coeff_pow_p g]; assumption
  /-
    case neg.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    m n : Nat
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max m n)) f) 0
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max m n)) g) 0
    hfg : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Max.max m n) 1)) ( …
    ⊢ Eq (PreTilt.valAux K v O hv p (HMul.hMul f g)) (HMul.hMul (PreTilt.valAux K  …
  -/
  rw [valAux_eq (coeff_add_ne_zero hm 1), valAux_eq (coeff_add_ne_zero hn 1), valAux_eq hfg]
  /-
    case neg.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    m n : Nat
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max m n)) f) 0
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max m n)) g) 0
    hfg : Ne ((Perfection.coeff (ModP K v O hv p) p (HAdd.hAdd (Max.max m n) 1)) ( …
    ⊢ Eq (HPow.hPow (ModP.preVal K v O hv p ((Perfection.coeff (ModP K v O hv p) p …
  -/
  rw [RingHom.map_mul] at hfg ⊢; rw [ModP.preVal_mul hfg, mul_pow]
                                 /-
                                   🎉 no goals
                                 -/


theorem valAux_add (f g : PreTilt K v O hv p) :
    valAux K v O hv p (f + g) ≤ max (valAux K v O hv p f) (valAux K v O hv p g) := by
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  by_cases hf : f = 0
    /-
      case pos
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      hf : Eq f 0
      ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
    -/
  · rw [hf, zero_add, valAux_zero, max_eq_right]; exact zero_le _
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  by_cases hg : g = 0
    /-
      case pos
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      hf : Not (Eq f 0)
      hg : Eq g 0
      ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
    -/
  · rw [hg, add_zero, valAux_zero, max_eq_left]; exact zero_le _
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  by_cases hfg : f + g = 0
    /-
      case pos
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      hf : Not (Eq f 0)
      hg : Not (Eq g 0)
      hfg : Eq (HAdd.hAdd f g) 0
      ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
    -/
  · rw [hfg, valAux_zero]; exact zero_le _
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Not (Eq f 0)
    hg : Not (Eq g 0)
    hfg : Not (Eq (HAdd.hAdd f g) 0)
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  replace hf : ∃ n, coeff _ _ n f ≠ 0 := not_forall.1 fun h => hf <| Perfection.ext h
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hg : Not (Eq g 0)
    hfg : Not (Eq (HAdd.hAdd f g) 0)
    hf : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  replace hg : ∃ n, coeff _ _ n g ≠ 0 := not_forall.1 fun h => hg <| Perfection.ext h
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hfg : Not (Eq (HAdd.hAdd f g) 0)
    hf : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    hg : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) g) 0
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  replace hfg : ∃ n, coeff _ _ n (f + g) ≠ 0 := not_forall.1 fun h => hfg <| Perfection.ext h
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    hf : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    hg : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) g) 0
    hfg : Exists fun n => Ne ((Perfection.coeff (ModP K v O hv p) p n) (HAdd.hAdd  …
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  obtain ⟨m, hm⟩ := hf; obtain ⟨n, hn⟩ := hg; obtain ⟨k, hk⟩ := hfg
  /-
    case neg.intro.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    m : Nat
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p m) f) 0
    n : Nat
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p n) g) 0
    k : Nat
    hk : Ne ((Perfection.coeff (ModP K v O hv p) p k) (HAdd.hAdd f g)) 0
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  replace hm := coeff_ne_zero_of_le hm (le_trans (le_max_left m n) (le_max_left _ k))
  /-
    case neg.intro.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    m n : Nat
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p n) g) 0
    k : Nat
    hk : Ne ((Perfection.coeff (ModP K v O hv p) p k) (HAdd.hAdd f g)) 0
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) f) 0
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  replace hn := coeff_ne_zero_of_le hn (le_trans (le_max_right m n) (le_max_left _ k))
  /-
    case neg.intro.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    m n k : Nat
    hk : Ne ((Perfection.coeff (ModP K v O hv p) p k) (HAdd.hAdd f g)) 0
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) f) 0
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) g) 0
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  replace hk := coeff_ne_zero_of_le hk (le_max_right (max m n) k)
  /-
    case neg.intro.intro.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f g : PreTilt K v O hv p
    m n k : Nat
    hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) f) 0
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) g) 0
    hk : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) (HAd …
    ⊢ LE.le (PreTilt.valAux K v O hv p (HAdd.hAdd f g)) (Max.max (PreTilt.valAux K …
  -/
  rw [valAux_eq hm, valAux_eq hn, valAux_eq hk, RingHom.map_add]
  cases' le_max_iff.1
      (ModP.preVal_add (coeff _ _ (max (max m n) k) f) (coeff _ _ (max (max m n) k) g)) with h h
    /-
      case neg.intro.intro.intro.inl
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      m n k : Nat
      hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) f) 0
      hn : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) g) 0
      hk : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) (HAd …
      h : LE.le (ModP.preVal K v O hv p (HAdd.hAdd ((Perfection.coeff (ModP K v O hv …
      ⊢ LE.le (HPow.hPow (ModP.preVal K v O hv p (HAdd.hAdd ((Perfection.coeff (ModP …
    -/
  · exact le_max_of_le_left (pow_le_pow_left' h _)
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.inr
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f g : PreTilt K v O hv p
      m n k : Nat
      hm : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) f) 0
      hn : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) g) 0
      hk : Ne ((Perfection.coeff (ModP K v O hv p) p (Max.max (Max.max m n) k)) (HAd …
      h : LE.le (ModP.preVal K v O hv p (HAdd.hAdd ((Perfection.coeff (ModP K v O hv …
      ⊢ LE.le (HPow.hPow (ModP.preVal K v O hv p (HAdd.hAdd ((Perfection.coeff (ModP …
    -/
  · exact le_max_of_le_right (pow_le_pow_left' h _)
    /-
      🎉 no goals
    -/


/-- The valuation `Perfection(O/(p)) → ℝ≥0`.
Given `f ∈ Perfection(O/(p))`, if `f = 0` then output `0`;
otherwise output `preVal(f(n))^(p^n)` for any `n` such that `f(n) ≠ 0`. -/
noncomputable def val : Valuation (PreTilt K v O hv p) ℝ≥0 where
  toFun := valAux K v O hv p
  map_one' := valAux_one
  map_mul' := valAux_mul
  map_zero' := valAux_zero
  map_add_le_max' := valAux_add


theorem map_eq_zero {f : PreTilt K v O hv p} : val K v O hv p f = 0 ↔ f = 0 := by
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    ⊢ Iff (Eq ((PreTilt.val K v O hv p) f) 0) (Eq f 0)
  -/
  by_cases hf0 : f = 0
    /-
      case pos
      K : Type u₁
      inst✝⁴ : Field K
      v : Valuation K NNReal
      O : Type u₂
      inst✝³ : CommRing O
      inst✝² : Algebra O K
      hv : v.Integers O
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Ne (v ↑p) 1)
      f : PreTilt K v O hv p
      hf0 : Eq f 0
      ⊢ Iff (Eq ((PreTilt.val K v O hv p) f) 0) (Eq f 0)
    -/
  · rw [hf0]; exact iff_of_true (Valuation.map_zero _) rfl
              /-
                🎉 no goals
              -/
  /-
    case neg
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    hf0 : Not (Eq f 0)
    ⊢ Iff (Eq ((PreTilt.val K v O hv p) f) 0) (Eq f 0)
  -/
  obtain ⟨n, hn⟩ : ∃ n, coeff _ _ n f ≠ 0 := not_forall.1 fun h => hf0 <| Perfection.ext h
  /-
    case neg.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    hf0 : Not (Eq f 0)
    n : Nat
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    ⊢ Iff (Eq ((PreTilt.val K v O hv p) f) 0) (Eq f 0)
  -/
  show valAux K v O hv p f = 0 ↔ f = 0; refine iff_of_false (fun hvf => hn ?_) hf0
  /-
    case neg.intro
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    f : PreTilt K v O hv p
    hf0 : Not (Eq f 0)
    n : Nat
    hn : Ne ((Perfection.coeff (ModP K v O hv p) p n) f) 0
    hvf : Eq (PreTilt.valAux K v O hv p f) 0
    ⊢ Eq ((Perfection.coeff (ModP K v O hv p) p n) f) 0
  -/
  rw [valAux_eq hn] at hvf; replace hvf := pow_eq_zero hvf; rwa [ModP.preVal_eq_zero] at hvf
                                                            /-
                                                              🎉 no goals
                                                            -/


instance [hp : Fact p.Prime] : IsDomain (PreTilt K v O hv p) := by
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    hp : Fact (Nat.Prime p)
    ⊢ IsDomain (PreTilt K v O hv p)
  -/
  haveI : Nontrivial (PreTilt K v O hv p) := ⟨(CharP.nontrivial_of_char_ne_one hp.1.ne_one).1⟩
  haveI : NoZeroDivisors (PreTilt K v O hv p) :=
    ⟨fun hfg => by
      simp_rw [← map_eq_zero] at hfg ⊢; contrapose! hfg; rw [Valuation.map_mul]
      exact mul_ne_zero hfg.1 hfg.2⟩
  /-
    K : Type u₁
    inst✝⁴ : Field K
    v : Valuation K NNReal
    O : Type u₂
    inst✝³ : CommRing O
    inst✝² : Algebra O K
    hv : v.Integers O
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Ne (v ↑p) 1)
    hp : Fact (Nat.Prime p)
    this✝ : Nontrivial (PreTilt K v O hv p)
    this : NoZeroDivisors (PreTilt K v O hv p)
    ⊢ IsDomain (PreTilt K v O hv p)
  -/
  exact NoZeroDivisors.to_isDomain _
  /-
    🎉 no goals
  -/


/-- The tilt of a field, as defined in Perfectoid Spaces by Peter Scholze, as in
[scholze2011perfectoid]. Given a field `K` with valuation `K → ℝ≥0` and ring of integers `O`,
this is implemented as the fraction field of the perfection of `O/(p)`. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
def Tilt [Fact p.Prime] [Fact (v p ≠ 1)] :=
  FractionRing (PreTilt K v O hv p)


noncomputable instance [Fact p.Prime] [Fact (v p ≠ 1)] : Field (Tilt K v O hv p) :=
  FractionRing.field _


