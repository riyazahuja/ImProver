/-- Conjugation-negation. Sends `f` to `fun x ↦ conj (f (-x))`. -/
def conjneg (f : G → R) : G → R := conj fun x ↦ f (-x)


@[simp] lemma conjneg_apply (f : G → R) (x : G) : conjneg f x = conj (f (-x)) := rfl

                                                                          /-
                                                                            G : Type u_2
                                                                            R : Type u_3
                                                                            inst✝² : AddGroup G
                                                                            inst✝¹ : CommSemiring R
                                                                            inst✝ : StarRing R
                                                                            f : G → R
                                                                            ⊢ Eq (conjneg (conjneg f)) f
                                                                          -/
@[simp] lemma conjneg_conjneg (f : G → R) : conjneg (conjneg f) = f := by ext; simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma conjneg_involutive : Involutive (conjneg : (G → R) → G → R) := conjneg_conjneg

lemma conjneg_bijective : Bijective (conjneg : (G → R) → G → R) := conjneg_involutive.bijective

lemma conjneg_injective : Injective (conjneg : (G → R) → G → R) := conjneg_involutive.injective

lemma conjneg_surjective : Surjective (conjneg : (G → R) → G → R) := conjneg_involutive.surjective


@[simp] lemma conjneg_inj : conjneg f = conjneg g ↔ f = g := conjneg_injective.eq_iff

lemma conjneg_ne_conjneg : conjneg f ≠ conjneg g ↔ f ≠ g := conjneg_injective.ne_iff


@[simp] lemma conjneg_conj (f : G → R) : conjneg (conj f) = conj (conjneg f) := rfl


                                                           /-
                                                             G : Type u_2
                                                             R : Type u_3
                                                             inst✝² : AddGroup G
                                                             inst✝¹ : CommSemiring R
                                                             inst✝ : StarRing R
                                                             ⊢ Eq (conjneg 0) 0
                                                           -/
@[simp] lemma conjneg_zero : conjneg (0 : G → R) = 0 := by ext; simp
                                                                /-
                                                                  🎉 no goals
                                                                -/

                                                          /-
                                                            G : Type u_2
                                                            R : Type u_3
                                                            inst✝² : AddGroup G
                                                            inst✝¹ : CommSemiring R
                                                            inst✝ : StarRing R
                                                            ⊢ Eq (conjneg 1) 1
                                                          -/
@[simp] lemma conjneg_one : conjneg (1 : G → R) = 1 := by ext; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/

                                                                                        /-
                                                                                          G : Type u_2
                                                                                          R : Type u_3
                                                                                          inst✝² : AddGroup G
                                                                                          inst✝¹ : CommSemiring R
                                                                                          inst✝ : StarRing R
                                                                                          f g : G → R
                                                                                          ⊢ Eq (conjneg (HAdd.hAdd f g)) (HAdd.hAdd (conjneg f) (conjneg g))
                                                                                        -/
@[simp] lemma conjneg_add (f g : G → R) : conjneg (f + g) = conjneg f + conjneg g := by ext; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/

                                                                                        /-
                                                                                          G : Type u_2
                                                                                          R : Type u_3
                                                                                          inst✝² : AddGroup G
                                                                                          inst✝¹ : CommSemiring R
                                                                                          inst✝ : StarRing R
                                                                                          f g : G → R
                                                                                          ⊢ Eq (conjneg (HMul.hMul f g)) (HMul.hMul (conjneg f) (conjneg g))
                                                                                        -/
@[simp] lemma conjneg_mul (f g : G → R) : conjneg (f * g) = conjneg f * conjneg g := by ext; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[simp] lemma conjneg_sum (s : Finset ι) (f : ι → G → R) :
                                                          /-
                                                            ι : Type u_1
                                                            G : Type u_2
                                                            R : Type u_3
                                                            inst✝² : AddGroup G
                                                            inst✝¹ : CommSemiring R
                                                            inst✝ : StarRing R
                                                            s : Finset ι
                                                            f : ι → G → R
                                                            ⊢ Eq (conjneg (s.sum fun i => f i)) (s.sum fun i => conjneg (f i))
                                                          -/
    conjneg (∑ i ∈ s, f i) = ∑ i ∈ s, conjneg (f i) := by ext; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp] lemma conjneg_prod (s : Finset ι) (f : ι → G → R) :
                                                          /-
                                                            ι : Type u_1
                                                            G : Type u_2
                                                            R : Type u_3
                                                            inst✝² : AddGroup G
                                                            inst✝¹ : CommSemiring R
                                                            inst✝ : StarRing R
                                                            s : Finset ι
                                                            f : ι → G → R
                                                            ⊢ Eq (conjneg (s.prod fun i => f i)) (s.prod fun i => conjneg (f i))
                                                          -/
    conjneg (∏ i ∈ s, f i) = ∏ i ∈ s, conjneg (f i) := by ext; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp] lemma conjneg_eq_zero : conjneg f = 0 ↔ f = 0 := by
  /-
    G : Type u_2
    R : Type u_3
    inst✝² : AddGroup G
    inst✝¹ : CommSemiring R
    inst✝ : StarRing R
    f : G → R
    ⊢ Iff (Eq (conjneg f) 0) (Eq f 0)
  -/
  rw [← conjneg_inj, conjneg_conjneg, conjneg_zero]
  /-
    🎉 no goals
  -/


@[simp] lemma conjneg_eq_one : conjneg f = 1 ↔ f = 1 := by
  /-
    G : Type u_2
    R : Type u_3
    inst✝² : AddGroup G
    inst✝¹ : CommSemiring R
    inst✝ : StarRing R
    f : G → R
    ⊢ Iff (Eq (conjneg f) 1) (Eq f 1)
  -/
  rw [← conjneg_inj, conjneg_conjneg, conjneg_one]
  /-
    🎉 no goals
  -/


lemma conjneg_ne_zero : conjneg f ≠ 0 ↔ f ≠ 0 := conjneg_eq_zero.not

lemma conjneg_ne_one : conjneg f ≠ 1 ↔ f ≠ 1 := conjneg_eq_one.not


lemma sum_conjneg [Fintype G] (f : G → R) : ∑ a, conjneg f a = ∑ a, conj (f a) :=
  Fintype.sum_equiv (Equiv.neg _) _ _ fun _ ↦ rfl


@[simp] lemma support_conjneg (f : G → R) : support (conjneg f) = -support f := by
  /-
    G : Type u_2
    R : Type u_3
    inst✝² : AddGroup G
    inst✝¹ : CommSemiring R
    inst✝ : StarRing R
    f : G → R
    ⊢ Eq (Function.support (conjneg f)) (Neg.neg (Function.support f))
  -/
  ext; simp [starRingEnd_apply]
       /-
         🎉 no goals
       -/


/-- `conjneg` bundled as a ring homomorphism. -/
@[simps] def conjnegRingHom : (G → R) →+* (G → R) where
  toFun := conjneg
  map_zero' := conjneg_zero
  map_one' := conjneg_one
  map_add' := conjneg_add
  map_mul' := conjneg_mul


                                                                                        /-
                                                                                          G : Type u_2
                                                                                          R : Type u_3
                                                                                          inst✝² : AddGroup G
                                                                                          inst✝¹ : CommRing R
                                                                                          inst✝ : StarRing R
                                                                                          f g : G → R
                                                                                          ⊢ Eq (conjneg (HSub.hSub f g)) (HSub.hSub (conjneg f) (conjneg g))
                                                                                        -/
@[simp] lemma conjneg_sub (f g : G → R) : conjneg (f - g) = conjneg f - conjneg g := by ext; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/

                                                                        /-
                                                                          G : Type u_2
                                                                          R : Type u_3
                                                                          inst✝² : AddGroup G
                                                                          inst✝¹ : CommRing R
                                                                          inst✝ : StarRing R
                                                                          f : G → R
                                                                          ⊢ Eq (conjneg (Neg.neg f)) (Neg.neg (conjneg f))
                                                                        -/
@[simp] lemma conjneg_neg (f : G → R) : conjneg (-f) = -conjneg f := by ext; simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


