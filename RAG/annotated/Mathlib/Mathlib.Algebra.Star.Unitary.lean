/-- In a *-monoid, `unitary R` is the submonoid consisting of all the elements `U` of
`R` such that `star U * U = 1` and `U * star U = 1`.
-/
def unitary (R : Type*) [Monoid R] [StarMul R] : Submonoid R where
  carrier := { U | star U * U = 1 ∧ U * star U = 1 }
                 /-
                   R : Type u_1
                   inst✝¹ : Monoid R
                   inst✝ : StarMul R
                   ⊢ Membership.mem { carrier := setOf fun U => And (Eq (HMul.hMul (Star.star U)  …
                 -/
  one_mem' := by simp only [mul_one, and_self_iff, Set.mem_setOf_eq, star_one]
    /-
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      U B : R
      x✝¹ : Membership.mem (setOf fun U => And (Eq (HMul.hMul (Star.star U) U) 1) (E …
      x✝ : Membership.mem (setOf fun U => And (Eq (HMul.hMul (Star.star U) U) 1) (Eq …
      hA₁ : Eq (HMul.hMul (Star.star U) U) 1
      hA₂ : Eq (HMul.hMul U (Star.star U)) 1
      hB₁ : Eq (HMul.hMul (Star.star B) B) 1
      hB₂ : Eq (HMul.hMul B (Star.star B)) 1
      ⊢ Membership.mem (setOf fun U => And (Eq (HMul.hMul (Star.star U) U) 1) (Eq (H …
    -/
                 /-
                   🎉 no goals
                 -/
  mul_mem' := @fun U B ⟨hA₁, hA₂⟩ ⟨hB₁, hB₂⟩ => by
    refine ⟨?_, ?_⟩
    · calc
        star (U * B) * (U * B) = star B * star U * U * B := by simp only [mul_assoc, star_mul]
        _ = star B * (star U * U) * B := by rw [← mul_assoc]
        _ = 1 := by rw [hA₁, mul_one, hB₁]
    · calc
        U * B * star (U * B) = U * B * (star B * star U) := by rw [star_mul]
        _ = U * (B * star B) * star U := by simp_rw [← mul_assoc]
        _ = 1 := by rw [hB₂, mul_one, hA₂]


theorem mem_iff {U : R} : U ∈ unitary R ↔ star U * U = 1 ∧ U * star U = 1 :=
  Iff.rfl


@[simp]
theorem star_mul_self_of_mem {U : R} (hU : U ∈ unitary R) : star U * U = 1 :=
  hU.1


@[simp]
theorem mul_star_self_of_mem {U : R} (hU : U ∈ unitary R) : U * star U = 1 :=
  hU.2


theorem star_mem {U : R} (hU : U ∈ unitary R) : star U ∈ unitary R :=
      /-
        R : Type u_1
        inst✝¹ : Monoid R
        inst✝ : StarMul R
        U : R
        hU : Membership.mem (unitary R) U
        ⊢ Eq (HMul.hMul (Star.star (Star.star U)) (Star.star U)) 1
      -/
      /-
        🎉 no goals
      -/
  ⟨by rw [star_star, mul_star_self_of_mem hU], by rw [star_star, star_mul_self_of_mem hU]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem star_mem_iff {U : R} : star U ∈ unitary R ↔ U ∈ unitary R :=
  ⟨fun h => star_star U ▸ star_mem h, star_mem⟩


instance : Star (unitary R) :=
  ⟨fun U => ⟨star U, star_mem U.prop⟩⟩


@[simp, norm_cast]
theorem coe_star {U : unitary R} : ↑(star U) = (star U : R) :=
  rfl


theorem coe_star_mul_self (U : unitary R) : (star U : R) * U = 1 :=
  star_mul_self_of_mem U.prop


theorem coe_mul_star_self (U : unitary R) : (U : R) * star U = 1 :=
  mul_star_self_of_mem U.prop


@[simp]
theorem star_mul_self (U : unitary R) : star U * U = 1 :=
  Subtype.ext <| coe_star_mul_self U


@[simp]
theorem mul_star_self (U : unitary R) : U * star U = 1 :=
  Subtype.ext <| coe_mul_star_self U


instance : Group (unitary R) :=
  { Submonoid.toMonoid _ with
    inv := star
    inv_mul_cancel := star_mul_self }


instance : InvolutiveStar (unitary R) :=
  ⟨by
    /-
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      ⊢ Function.Involutive Star.star
    -/
    intro x
    /-
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      x : Subtype fun x => Membership.mem (unitary R) x
      ⊢ Eq (Star.star (Star.star x)) x
    -/
    ext
    /-
      case a
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      x : Subtype fun x => Membership.mem (unitary R) x
      ⊢ Eq ↑(Star.star (Star.star x)) ↑x
    -/
    rw [coe_star, coe_star, star_star]⟩
    /-
      🎉 no goals
    -/


instance : StarMul (unitary R) :=
  ⟨by
    /-
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      ⊢ ∀ (r s : Subtype fun x => Membership.mem (unitary R) x), Eq (Star.star (HMul …
    -/
    intro x y
    /-
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      x y : Subtype fun x => Membership.mem (unitary R) x
      ⊢ Eq (Star.star (HMul.hMul x y)) (HMul.hMul (Star.star y) (Star.star x))
    -/
    ext
    /-
      case a
      R : Type u_1
      inst✝¹ : Monoid R
      inst✝ : StarMul R
      x y : Subtype fun x => Membership.mem (unitary R) x
      ⊢ Eq ↑(Star.star (HMul.hMul x y)) ↑(HMul.hMul (Star.star y) (Star.star x))
    -/
    rw [coe_star, Submonoid.coe_mul, Submonoid.coe_mul, coe_star, coe_star, star_mul]⟩
    /-
      🎉 no goals
    -/


instance : Inhabited (unitary R) :=
  ⟨1⟩


theorem star_eq_inv (U : unitary R) : star U = U⁻¹ :=
  rfl


theorem star_eq_inv' : (star : unitary R → unitary R) = Inv.inv :=
  rfl


/-- The unitary elements embed into the units. -/
@[simps]
def toUnits : unitary R →* Rˣ where
  toFun x := ⟨x, ↑x⁻¹, coe_mul_star_self x, coe_star_mul_self x⟩
  map_one' := Units.ext rfl
  map_mul' _ _ := Units.ext rfl


theorem toUnits_injective : Function.Injective (toUnits : unitary R → Rˣ) := fun _ _ h =>
  Subtype.ext <| Units.ext_iff.mp h


theorem _root_.IsUnit.mem_unitary_of_star_mul_self  {u : R} (hu : IsUnit u)
    (h_mul : star u * u = 1) : u ∈ unitary R := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    inst✝ : StarMul R
    u : R
    hu : IsUnit u
    h_mul : Eq (HMul.hMul (Star.star u) u) 1
    ⊢ Membership.mem (unitary R) u
  -/
  refine unitary.mem_iff.mpr ⟨h_mul, ?_⟩
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    inst✝ : StarMul R
    u : R
    hu : IsUnit u
    h_mul : Eq (HMul.hMul (Star.star u) u) 1
    ⊢ Eq (HMul.hMul u (Star.star u)) 1
  -/
  lift u to Rˣ using hu
  /-
    case intro
    R : Type u_1
    inst✝¹ : Monoid R
    inst✝ : StarMul R
    u : Units R
    h_mul : Eq (HMul.hMul (Star.star ↑u) ↑u) 1
    ⊢ Eq (HMul.hMul (↑u) (Star.star ↑u)) 1
  -/
  exact left_inv_eq_right_inv h_mul u.mul_inv ▸ u.mul_inv
  /-
    🎉 no goals
  -/


theorem _root_.IsUnit.mem_unitary_of_mul_star_self {u : R} (hu : IsUnit u)
    (h_mul : u * star u = 1) : u ∈ unitary R :=
  star_star u ▸
    (hu.star.mem_unitary_of_star_mul_self ((star_star u).symm ▸ h_mul) |> unitary.star_mem)


instance instIsStarNormal (u : unitary R) : IsStarNormal u where
  star_comm_self := star_mul_self u |>.trans <| (mul_star_self u).symm


instance coe_isStarNormal (u : unitary R) : IsStarNormal (u : R) where
  star_comm_self := congr(Subtype.val $(star_comm_self' u))


lemma _root_.isStarNormal_of_mem_unitary {u : R} (hu : u ∈ unitary R) : IsStarNormal u :=
  coe_isStarNormal ⟨u, hu⟩


lemma map_mem {r : R} (hr : r ∈ unitary R) : f r ∈ unitary S := by
  /-
    F : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝⁶ : Monoid R
    inst✝⁵ : StarMul R
    inst✝⁴ : Monoid S
    inst✝³ : StarMul S
    inst✝² : FunLike F R S
    inst✝¹ : StarHomClass F R S
    inst✝ : MonoidHomClass F R S
    f : F
    r : R
    hr : Membership.mem (unitary R) r
    ⊢ Membership.mem (unitary S) (f r)
  -/
  rw [unitary.mem_iff] at hr
  /-
    F : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝⁶ : Monoid R
    inst✝⁵ : StarMul R
    inst✝⁴ : Monoid S
    inst✝³ : StarMul S
    inst✝² : FunLike F R S
    inst✝¹ : StarHomClass F R S
    inst✝ : MonoidHomClass F R S
    f : F
    r : R
    hr : And (Eq (HMul.hMul (Star.star r) r) 1) (Eq (HMul.hMul r (Star.star r)) 1)
    ⊢ Membership.mem (unitary S) (f r)
  -/
  simpa [map_star, map_mul] using And.intro congr(f $(hr.1)) congr(f $(hr.2))
  /-
    🎉 no goals
  -/


/-- The group homomorphism between unitary subgroups of star monoids induced by a star
homomorphism -/
@[simps]
def map : unitary R →* unitary S where
  toFun := Subtype.map f (fun _ ↦ map_mem f)
  map_one' := Subtype.ext <| map_one f
  map_mul' _ _ := Subtype.ext <| map_mul f _ _


                                                                                 /-
                                                                                   F : Type u_2
                                                                                   R : Type u_3
                                                                                   S : Type u_4
                                                                                   inst✝⁶ : Monoid R
                                                                                   inst✝⁵ : StarMul R
                                                                                   inst✝⁴ : Monoid S
                                                                                   inst✝³ : StarMul S
                                                                                   inst✝² : FunLike F R S
                                                                                   inst✝¹ : StarHomClass F R S
                                                                                   inst✝ : MonoidHomClass F R S
                                                                                   f : F
                                                                                   ⊢ Eq (unitary.toUnits.comp (unitary.map f)) ((Units.map ↑f).comp unitary.toUni …
                                                                                 -/
lemma toUnits_comp_map : toUnits.comp (map f) = (Units.map f).comp toUnits := by ext; rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


instance : CommGroup (unitary R) :=
  { inferInstanceAs (Group (unitary R)), Submonoid.toCommMonoid _ with }


theorem mem_iff_star_mul_self {U : R} : U ∈ unitary R ↔ star U * U = 1 :=
  mem_iff.trans <| and_iff_left_of_imp fun h => mul_comm (star U) U ▸ h


theorem mem_iff_self_mul_star {U : R} : U ∈ unitary R ↔ U * star U = 1 :=
  mem_iff.trans <| and_iff_right_of_imp fun h => mul_comm U (star U) ▸ h


@[norm_cast]
theorem coe_inv (U : unitary R) : ↑U⁻¹ = (U⁻¹ : R) :=
  eq_inv_of_mul_eq_one_right <| coe_mul_star_self _


@[norm_cast]
theorem coe_div (U₁ U₂ : unitary R) : ↑(U₁ / U₂) = (U₁ / U₂ : R) := by
  /-
    R : Type u_1
    inst✝¹ : GroupWithZero R
    inst✝ : StarMul R
    U₁ U₂ : Subtype fun x => Membership.mem (unitary R) x
    ⊢ Eq (↑(HDiv.hDiv U₁ U₂)) (HDiv.hDiv ↑U₁ ↑U₂)
  -/
  simp only [div_eq_mul_inv, coe_inv, Submonoid.coe_mul]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_zpow (U : unitary R) (z : ℤ) : ↑(U ^ z) = (U : R) ^ z := by
  /-
    R : Type u_1
    inst✝¹ : GroupWithZero R
    inst✝ : StarMul R
    U : Subtype fun x => Membership.mem (unitary R) x
    z : Int
    ⊢ Eq (↑(HPow.hPow U z)) (HPow.hPow (↑U) z)
  -/
  induction z
    /-
      case ofNat
      R : Type u_1
      inst✝¹ : GroupWithZero R
      inst✝ : StarMul R
      U : Subtype fun x => Membership.mem (unitary R) x
      a✝ : Nat
      ⊢ Eq (↑(HPow.hPow U (Int.ofNat a✝))) (HPow.hPow (↑U) (Int.ofNat a✝))
    -/
  · simp [SubmonoidClass.coe_pow]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      R : Type u_1
      inst✝¹ : GroupWithZero R
      inst✝ : StarMul R
      U : Subtype fun x => Membership.mem (unitary R) x
      a✝ : Nat
      ⊢ Eq (↑(HPow.hPow U (Int.negSucc a✝))) (HPow.hPow (↑U) (Int.negSucc a✝))
    -/
  · simp [coe_inv]
    /-
      🎉 no goals
    -/


instance : Neg (unitary R) where
  neg U :=
            /-
              R : Type u_1
              inst✝¹ : Ring R
              inst✝ : StarRing R
              U : Subtype fun x => Membership.mem (unitary R) x
              ⊢ Membership.mem (unitary R) (Neg.neg ↑U)
            -/
    ⟨-U, by simp [mem_iff, star_neg, neg_mul_neg]⟩
            /-
              🎉 no goals
            -/


@[norm_cast]
theorem coe_neg (U : unitary R) : ↑(-U) = (-U : R) :=
  rfl


instance : HasDistribNeg (unitary R) :=
  Subtype.coe_injective.hasDistribNeg _ coe_neg (unitary R).coe_mul


/-- Unitary conjugation preserves the spectrum, star on left. -/
@[simp]
lemma spectrum.unitary_conjugate {a : A} {u : unitary A} :
    spectrum R (u * a * (star u : A)) = spectrum R a :=
  spectrum.units_conjugate (u := unitary.toUnits u)


/-- Unitary conjugation preserves the spectrum, star on right. -/
@[simp]
lemma spectrum.unitary_conjugate' {a : A} {u : unitary A} :
    spectrum R ((star u : A) * a * u) = spectrum R a := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝³ : CommSemiring R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : StarMul A
    a : A
    u : Subtype fun x => Membership.mem (unitary A) x
    ⊢ Eq (spectrum R (HMul.hMul (HMul.hMul (Star.star ↑u) a) ↑u)) (spectrum R a)
  -/
  simpa using spectrum.unitary_conjugate (u := star u)
  /-
    🎉 no goals
  -/


