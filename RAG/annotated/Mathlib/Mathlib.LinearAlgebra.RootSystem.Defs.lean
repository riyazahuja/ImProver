/-- Given two perfectly-paired `R`-modules `M` and `N`, a root pairing with indexing set `ι`
is the data of an `ι`-indexed subset of `M` ("the roots"), an `ι`-indexed subset of `N`
("the coroots"), and an `ι`-indexed set of permutations of `ι`, such that each root-coroot pair
evaluates to `2`, and the permutation attached to each element of `ι` is compatible with the
reflections on the corresponding roots and coroots.

It exists to allow for a convenient unification of the theories of root systems and root data. -/
structure RootPairing extends PerfectPairing R M N where
  /-- A parametrized family of vectors, called roots. -/
  root : ι ↪ M
  /-- A parametrized family of dual vectors, called coroots. -/
  coroot : ι ↪ N
  root_coroot_two : ∀ i, toLin (root i) (coroot i) = 2
  /-- A parametrized family of permutations, induced by reflections. This corresponds to the
      classical requirement that the symmetry attached to each root (later defined in
      `RootPairing.reflection`) leave the whole set of roots stable: as explained above, we
      formalize this stability by fixing the image of the roots through each reflection (whence the
      permutation); and similarly for coroots. -/
  reflection_perm : ι → (ι ≃ ι)
  reflection_perm_root : ∀ i j,
    root j - toPerfectPairing (root j) (coroot i) • root i = root (reflection_perm i j)
  reflection_perm_coroot : ∀ i j,
    coroot j - toPerfectPairing (root i) (coroot j) • coroot i = coroot (reflection_perm i j)


/-- A root datum is a root pairing with coefficients in the integers and for which the root and
coroot spaces are finitely-generated free Abelian groups.

Note that the latter assumptions `[Free ℤ X₁] [Finite ℤ X₁] [Free ℤ X₂] [Finite ℤ X₂]` should be
supplied as mixins. -/
abbrev RootDatum (X₁ X₂ : Type*) [AddCommGroup X₁] [AddCommGroup X₂] := RootPairing ι ℤ X₁ X₂


/-- A root system is a root pairing for which the roots span their ambient module.

Note that this is slightly more general than the usual definition in the sense that `N` is not
required to be the dual of `M`. -/
structure RootSystem extends RootPairing ι R M N where
  span_eq_top : span R (range root) = ⊤


lemma ne_zero [CharZero R] : (P.root i : M) ≠ 0 :=
             /-
               ι : Type u_1
               R : Type u_2
               M : Type u_3
               N : Type u_4
               inst✝⁵ : CommRing R
               inst✝⁴ : AddCommGroup M
               inst✝³ : Module R M
               inst✝² : AddCommGroup N
               inst✝¹ : Module R N
               P : RootPairing ι R M N
               i : ι
               inst✝ : CharZero R
               h : Eq (P.root i) 0
               ⊢ False
             -/
  fun h ↦ by simpa [h, map_zero] using P.root_coroot_two i
             /-
               🎉 no goals
             -/


lemma ne_zero' [CharZero R] : (P.coroot i : N) ≠ 0 :=
             /-
               ι : Type u_1
               R : Type u_2
               M : Type u_3
               N : Type u_4
               inst✝⁵ : CommRing R
               inst✝⁴ : AddCommGroup M
               inst✝³ : Module R M
               inst✝² : AddCommGroup N
               inst✝¹ : Module R N
               P : RootPairing ι R M N
               i : ι
               inst✝ : CharZero R
               h : Eq (P.coroot i) 0
               ⊢ False
             -/
  fun h ↦ by simpa [h] using P.root_coroot_two i
             /-
               🎉 no goals
             -/


@[simp]
lemma toLin_toPerfectPairing (x : M) (y : N) : P.toLin x y = P.toPerfectPairing x y :=
  rfl


/-- If we interchange the roles of `M` and `N`, we still have a root pairing. -/
protected def flip : RootPairing ι R N M :=
  { P.toPerfectPairing.flip with
    root := P.coroot
    coroot := P.root
    root_coroot_two := P.root_coroot_two
    reflection_perm := P.reflection_perm
    reflection_perm_root := P.reflection_perm_coroot
    reflection_perm_coroot := P.reflection_perm_root }


@[simp]
lemma flip_flip : P.flip.flip = P :=
  rfl


/-- Roots written as functionals on the coweight space. -/
abbrev root' (i : ι) : Dual R N := P.toPerfectPairing (P.root i)


/-- Coroots written as functionals on the weight space. -/
abbrev coroot' (i : ι) : Dual R M := P.toPerfectPairing.flip (P.coroot i)


/-- This is the pairing between roots and coroots. -/
def pairing : R := P.root' i (P.coroot j)


@[simp]
lemma root_coroot_eq_pairing : P.toPerfectPairing (P.root i) (P.coroot j) = P.pairing i j :=
  rfl


@[simp]
lemma root'_coroot_eq_pairing : P.root' i (P.coroot j) = P.pairing i j :=
  rfl


@[simp]
lemma root_coroot'_eq_pairing : P.coroot' i (P.root j) = P.pairing j i :=
  rfl


lemma coroot_root_eq_pairing : P.toLin.flip (P.coroot i) (P.root j) = P.pairing j i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq ((P.toLin.flip (P.coroot i)) (P.root j)) (P.pairing j i)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma pairing_same : P.pairing i i = 2 := P.root_coroot_two i


lemma coroot_root_two :
    P.toLin.flip (P.coroot i) (P.root i) = 2 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ Eq ((P.toLin.flip (P.coroot i)) (P.root i)) 2
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The reflection associated to a root. -/
def reflection : M ≃ₗ[R] M :=
  Module.reflection (P.flip.root_coroot_two i)


@[simp]
lemma root_reflection_perm (j : ι) :
    P.root (P.reflection_perm i j) = (P.reflection i) (P.root j) :=
  (P.reflection_perm_root i j).symm


theorem mapsTo_reflection_root :
    MapsTo (P.reflection i) (range P.root) (range P.root) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ Set.MapsTo (⇑(P.reflection i)) (Set.range ⇑P.root) (Set.range ⇑P.root)
  -/
  rintro - ⟨j, rfl⟩
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Membership.mem (Set.range ⇑P.root) ((P.reflection i) (P.root j))
  -/
  exact P.root_reflection_perm i j ▸ mem_range_self (P.reflection_perm i j)
  /-
    🎉 no goals
  -/


lemma reflection_apply (x : M) :
    P.reflection i x = x - (P.coroot' i x) • P.root i :=
  rfl


lemma reflection_apply_root :
    P.reflection i (P.root j) = P.root j - (P.pairing j i) • P.root i :=
  rfl


@[simp]
lemma reflection_apply_self :
    P.reflection i (P.root i) = - P.root i :=
  Module.reflection_apply_self (P.coroot_root_two i)


@[simp]
lemma reflection_same (x : M) :
    P.reflection i (P.reflection i x) = x :=
  Module.involutive_reflection (P.coroot_root_two i) x


@[simp]
lemma reflection_inv :
    (P.reflection i)⁻¹ = P.reflection i :=
  rfl


@[simp]
lemma reflection_sq :
    P.reflection i ^ 2 = 1 :=
  mul_eq_one_iff_eq_inv.mpr rfl


@[simp]
lemma reflection_perm_sq :
    P.reflection_perm i ^ 2 = 1 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ Eq (HPow.hPow (P.reflection_perm i) 2) 1
  -/
  ext j
  /-
    case H
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq ((HPow.hPow (P.reflection_perm i) 2) j) (1 j)
  -/
  apply P.root.injective
  /-
    case H.a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq (P.root ((HPow.hPow (P.reflection_perm i) 2) j)) (P.root (1 j))
  -/
  simp only [sq, Equiv.Perm.mul_apply, root_reflection_perm, reflection_same, Equiv.Perm.one_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma reflection_perm_inv :
    (P.reflection_perm i)⁻¹ = P.reflection_perm i :=
  (mul_eq_one_iff_eq_inv.mp <| P.reflection_perm_sq i).symm


@[simp]
lemma reflection_perm_self : P.reflection_perm i (P.reflection_perm i j) = j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq ((P.reflection_perm i) ((P.reflection_perm i) j)) j
  -/
  apply P.root.injective
  /-
    case a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq (P.root ((P.reflection_perm i) ((P.reflection_perm i) j))) (P.root j)
  -/
  simp only [root_reflection_perm, reflection_same]
  /-
    🎉 no goals
  -/


lemma reflection_perm_involutive : Involutive (P.reflection_perm i) :=
                                      /-
                                        ι : Type u_1
                                        R : Type u_2
                                        M : Type u_3
                                        N : Type u_4
                                        inst✝⁴ : CommRing R
                                        inst✝³ : AddCommGroup M
                                        inst✝² : Module R M
                                        inst✝¹ : AddCommGroup N
                                        inst✝ : Module R N
                                        P : RootPairing ι R M N
                                        i : ι
                                        ⊢ Eq (Nat.iterate (⇑(P.reflection_perm i)) 2) id
                                      -/
  involutive_iff_iter_2_eq_id.mpr (by ext; simp)
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
lemma reflection_perm_symm : (P.reflection_perm i).symm = P.reflection_perm i :=
  Involutive.symm_eq_self_of_involutive (P.reflection_perm i) <| P.reflection_perm_involutive i


lemma bijOn_reflection_root :
    BijOn (P.reflection i) (range P.root) (range P.root) :=
  Module.bijOn_reflection_of_mapsTo _ <| P.mapsTo_reflection_root i


@[simp]
lemma reflection_image_eq :
    P.reflection i '' (range P.root) = range P.root :=
  (P.bijOn_reflection_root i).image_eq


/-- The reflection associated to a coroot. -/
def coreflection : N ≃ₗ[R] N :=
  Module.reflection (P.root_coroot_two i)


@[simp]
lemma coroot_reflection_perm (j : ι) :
    P.coroot (P.reflection_perm i j) = (P.coreflection i) (P.coroot j) :=
  (P.reflection_perm_coroot i j).symm


theorem mapsTo_coreflection_coroot :
    MapsTo (P.coreflection i) (range P.coroot) (range P.coroot) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ Set.MapsTo (⇑(P.coreflection i)) (Set.range ⇑P.coroot) (Set.range ⇑P.coroot)
  -/
  rintro - ⟨j, rfl⟩
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Membership.mem (Set.range ⇑P.coroot) ((P.coreflection i) (P.coroot j))
  -/
  exact P.coroot_reflection_perm i j ▸ mem_range_self (P.reflection_perm i j)
  /-
    🎉 no goals
  -/


lemma coreflection_apply (f : N) :
    P.coreflection i f = f - (P.root' i) f • P.coroot i :=
  rfl


lemma coreflection_apply_coroot :
    P.coreflection i (P.coroot j) = P.coroot j - (P.pairing i j) • P.coroot i :=
  rfl


@[simp]
lemma coreflection_apply_self :
    P.coreflection i (P.coroot i) = - P.coroot i :=
  Module.reflection_apply_self (P.flip.coroot_root_two i)


@[simp]
lemma coreflection_same (x : N) :
    P.coreflection i (P.coreflection i x) = x :=
  Module.involutive_reflection (P.flip.coroot_root_two i) x


@[simp]
lemma coreflection_inv :
    (P.coreflection i)⁻¹ = P.coreflection i :=
  rfl


@[simp]
lemma coreflection_sq :
    P.coreflection i ^ 2 = 1 :=
  mul_eq_one_iff_eq_inv.mpr rfl


lemma bijOn_coreflection_coroot : BijOn (P.coreflection i) (range P.coroot) (range P.coroot) :=
  bijOn_reflection_root P.flip i


@[simp]
lemma coreflection_image_eq :
    P.coreflection i '' (range P.coroot) = range P.coroot :=
  (P.bijOn_coreflection_coroot i).image_eq


lemma coreflection_eq_flip_reflection :
    P.coreflection i = P.flip.reflection i :=
  rfl


lemma reflection_dualMap_eq_coreflection :
    (P.reflection i).dualMap ∘ₗ P.toLin.flip = P.toLin.flip ∘ₗ P.coreflection i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ Eq ((↑(P.reflection i).dualMap).comp P.toLin.flip) (P.toLin.flip.comp ↑(P.co …
  -/
  ext n m
  /-
    case h.h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    n : N
    m : M
    ⊢ Eq ((((↑(P.reflection i).dualMap).comp P.toLin.flip) n) m) (((P.toLin.flip.c …
  -/
  simp [map_sub, coreflection_apply, reflection_apply, mul_comm (P.toPerfectPairing m (P.coroot i))]
  /-
    🎉 no goals
  -/


lemma coroot_eq_coreflection_of_root_eq
    {i j k : ι} (hk : P.root k = P.reflection i (P.root j)) :
    P.coroot k = P.coreflection i (P.coroot j) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j k : ι
    hk : Eq (P.root k) ((P.reflection i) (P.root j))
    ⊢ Eq (P.coroot k) ((P.coreflection i) (P.coroot j))
  -/
  rw [← P.root_reflection_perm, EmbeddingLike.apply_eq_iff_eq] at hk
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j k : ι
    hk : Eq k ((P.reflection_perm i) j)
    ⊢ Eq (P.coroot k) ((P.coreflection i) (P.coroot j))
  -/
  rw [← P.coroot_reflection_perm, hk]
  /-
    🎉 no goals
  -/


lemma coroot'_reflection_perm {i j : ι} :
    P.coroot' (P.reflection_perm i j) = P.coroot' j ∘ₗ P.reflection i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq (P.coroot' ((P.reflection_perm i) j)) (LinearMap.comp (P.coroot' j) ↑(P.r …
  -/
  ext y
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    y : M
    ⊢ Eq ((P.coroot' ((P.reflection_perm i) j)) y) ((LinearMap.comp (P.coroot' j)  …
  -/
  simp [coreflection_apply_coroot, reflection_apply, map_sub, mul_comm]
  /-
    🎉 no goals
  -/


lemma coroot'_reflection {i j : ι} (y : M) :
    P.coroot' j (P.reflection i y) = P.coroot' (P.reflection_perm i j) y :=
  (LinearMap.congr_fun P.coroot'_reflection_perm y).symm


lemma pairing_reflection_perm (i j k : ι) :
    P.pairing j (P.reflection_perm i k) = P.pairing (P.reflection_perm i j) k := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j k : ι
    ⊢ Eq (P.pairing j ((P.reflection_perm i) k)) (P.pairing ((P.reflection_perm i) …
  -/
  simp only [pairing, root', coroot_reflection_perm, root_reflection_perm]
  simp only [coreflection_apply_coroot, map_sub, map_smul, smul_eq_mul,
    reflection_apply_root]
  simp only [← toLin_toPerfectPairing, map_smul, LinearMap.smul_apply, map_sub, map_smul,
    LinearMap.sub_apply, smul_eq_mul]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j k : ι
    ⊢ Eq (HSub.hSub ((P.toLin (P.root j)) (P.coroot k)) (HMul.hMul (P.pairing i k) …
  -/
  simp only [PerfectPairing.toLin_apply, root'_coroot_eq_pairing, sub_right_inj, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
lemma pairing_reflection_perm_self_left (P : RootPairing ι R M N) (i j : ι) :
    P.pairing (P.reflection_perm i i) j = - P.pairing i j := by
  rw [pairing, root', ← reflection_perm_root, root'_coroot_eq_pairing, pairing_same, two_smul,
    sub_add_cancel_left, ← toLin_toPerfectPairing, LinearMap.map_neg₂, toLin_toPerfectPairing,
    root'_coroot_eq_pairing]


@[simp]
lemma pairing_reflection_perm_self_right (i j : ι) :
    P.pairing i (P.reflection_perm j j) = - P.pairing i j := by
  rw [pairing, ← reflection_perm_coroot, root_coroot_eq_pairing, pairing_same, two_smul,
    sub_add_cancel_left, ← toLin_toPerfectPairing, map_neg, toLin_toPerfectPairing,
    root_coroot_eq_pairing]


/-- A root pairing is said to be crystallographic if the pairing between a root and coroot is
always an integer. -/
class IsCrystallographic : Prop where
  exists_int : ∀ i j, ∃ z : ℤ, z = P.pairing i j


protected lemma exists_int [P.IsCrystallographic] (i j : ι) :
    ∃ z : ℤ, z = P.pairing i j :=
  IsCrystallographic.exists_int i j


lemma isCrystallographic_iff :
    P.IsCrystallographic ↔ ∀ i j, ∃ z : ℤ, z = P.pairing i j :=
  ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩


instance [P.IsCrystallographic] : P.flip.IsCrystallographic := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    i j : ι
    inst✝ : P.IsCrystallographic
    ⊢ P.flip.IsCrystallographic
  -/
  rw [isCrystallographic_iff, forall_comm]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    i j : ι
    inst✝ : P.IsCrystallographic
    ⊢ ∀ (b a : ι), Exists fun z => Eq (↑z) (P.flip.pairing a b)
  -/
  exact P.exists_int
  /-
    🎉 no goals
  -/


/-- A root pairing is said to be reduced if any linearly dependent pair of roots is related by a
sign. -/
def IsReduced : Prop :=
  ∀ i j, ¬ LinearIndependent R ![P.root i, P.root j] → (P.root i = P.root j ∨ P.root i = - P.root j)


lemma isReduced_iff : P.IsReduced ↔ ∀ i j : ι, i ≠ j →
    ¬ LinearIndependent R ![P.root i, P.root j] → P.root i = - P.root j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Iff P.IsReduced (∀ (i j : ι), Ne i j → Not (LinearIndependent R (Matrix.vecC …
  -/
  rw [IsReduced]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Iff (∀ (i j : ι), Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matri …
  -/
  refine ⟨fun h i j hij hLin ↦ ?_, fun h i j hLin  ↦ ?_⟩
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      h : ∀ (i j : ι), Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.v …
      i j : ι
      hij : Ne i j
      hLin : Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P. …
      ⊢ Eq (P.root i) (Neg.neg (P.root j))
    -/
  · specialize h i j hLin
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      hij : Ne i j
      hLin : Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P. …
      h : Or (Eq (P.root i) (P.root j)) (Eq (P.root i) (Neg.neg (P.root j)))
      ⊢ Eq (P.root i) (Neg.neg (P.root j))
    -/
    simp_all only [ne_eq, EmbeddingLike.apply_eq_iff_eq, false_or]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      h : ∀ (i j : ι), Ne i j → Not (LinearIndependent R (Matrix.vecCons (P.root i)  …
      i j : ι
      hLin : Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P. …
      ⊢ Or (Eq (P.root i) (P.root j)) (Eq (P.root i) (Neg.neg (P.root j)))
    -/
  · by_cases h' : i = j
      /-
        case pos
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        P : RootPairing ι R M N
        h : ∀ (i j : ι), Ne i j → Not (LinearIndependent R (Matrix.vecCons (P.root i)  …
        i j : ι
        hLin : Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P. …
        h' : Eq i j
        ⊢ Or (Eq (P.root i) (P.root j)) (Eq (P.root i) (Neg.neg (P.root j)))
      -/
    · exact Or.inl (congrArg P.root h')
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        P : RootPairing ι R M N
        h : ∀ (i j : ι), Ne i j → Not (LinearIndependent R (Matrix.vecCons (P.root i)  …
        i j : ι
        hLin : Not (LinearIndependent R (Matrix.vecCons (P.root i) (Matrix.vecCons (P. …
        h' : Not (Eq i j)
        ⊢ Or (Eq (P.root i) (P.root j)) (Eq (P.root i) (Neg.neg (P.root j)))
      -/
    · exact Or.inr (h i j h' hLin)
      /-
        🎉 no goals
      -/


/-- The linear span of roots. -/
abbrev rootSpan := span R (range P.root)


/-- The linear span of coroots. -/
abbrev corootSpan := span R (range P.coroot)


lemma coe_rootSpan_dualAnnihilator_map :
    P.rootSpan.dualAnnihilator.map P.toDualRight.symm = {x | ∀ i, P.root' i x = 0} := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq (↑(Submodule.map P.toDualRight.symm P.rootSpan.dualAnnihilator)) (setOf f …
  -/
  ext x
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : N
    ⊢ Iff (Membership.mem (↑(Submodule.map P.toDualRight.symm P.rootSpan.dualAnnih …
  -/
  rw [rootSpan, Submodule.map_coe, Submodule.coe_dualAnnihilator_span]
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : N
    ⊢ Iff (Membership.mem (Set.image (⇑P.toDualRight.symm) (setOf fun f => HasSubs …
  -/
  change x ∈ P.toDualRight.toEquiv.symm '' _ ↔ _
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : N
    ⊢ Iff (Membership.mem (Set.image (⇑P.toDualRight.toEquiv.symm) (setOf fun f => …
  -/
  rw [← Equiv.setOf_apply_symm_eq_image_setOf, Equiv.symm_symm]
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : N
    ⊢ Iff (Membership.mem (setOf fun b => HasSubset.Subset (Set.range ⇑P.root) ↑(L …
  -/
  simp [Set.range_subset_iff]
  /-
    🎉 no goals
  -/


lemma coe_corootSpan_dualAnnihilator_map :
    P.corootSpan.dualAnnihilator.map P.toDualLeft.symm = {x | ∀ i, P.coroot' i x = 0} :=
  P.flip.coe_rootSpan_dualAnnihilator_map


lemma rootSpan_dualAnnihilator_map_eq :
    P.rootSpan.dualAnnihilator.map P.toDualRight.symm =
      (span R (range P.root')).dualCoannihilator := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq (Submodule.map P.toDualRight.symm P.rootSpan.dualAnnihilator) (Submodule. …
  -/
  apply SetLike.coe_injective
  /-
    case a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq ↑(Submodule.map P.toDualRight.symm P.rootSpan.dualAnnihilator) ↑(Submodul …
  -/
  rw [Submodule.coe_dualCoannihilator_span, coe_rootSpan_dualAnnihilator_map]
  /-
    case a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq (setOf fun x => ∀ (i : ι), Eq ((P.root' i) x) 0) (setOf fun x => ∀ (f : M …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma corootSpan_dualAnnihilator_map_eq :
    P.corootSpan.dualAnnihilator.map P.toDualLeft.symm =
      (span R (range P.coroot')).dualCoannihilator :=
  P.flip.rootSpan_dualAnnihilator_map_eq


lemma mem_range_root_of_mem_range_reflection_of_mem_range_root
    {r : M ≃ₗ[R] M} {α : M} (hr : r ∈ range P.reflection) (hα : α ∈ range P.root) :
    r • α ∈ range P.root := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    r : LinearEquiv (RingHom.id R) M M
    α : M
    hr : Membership.mem (Set.range P.reflection) r
    hα : Membership.mem (Set.range ⇑P.root) α
    ⊢ Membership.mem (Set.range ⇑P.root) (HSMul.hSMul r α)
  -/
  obtain ⟨i, rfl⟩ := hr
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    α : M
    hα : Membership.mem (Set.range ⇑P.root) α
    i : ι
    ⊢ Membership.mem (Set.range ⇑P.root) (HSMul.hSMul (P.reflection i) α)
  -/
  obtain ⟨j, rfl⟩ := hα
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Membership.mem (Set.range ⇑P.root) (HSMul.hSMul (P.reflection i) (P.root j))
  -/
  exact ⟨P.reflection_perm i j, P.root_reflection_perm i j⟩
  /-
    🎉 no goals
  -/


lemma mem_range_coroot_of_mem_range_coreflection_of_mem_range_coroot
    {r : N ≃ₗ[R] N} {α : N} (hr : r ∈ range P.coreflection) (hα : α ∈ range P.coroot) :
    r • α ∈ range P.coroot := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    r : LinearEquiv (RingHom.id R) N N
    α : N
    hr : Membership.mem (Set.range P.coreflection) r
    hα : Membership.mem (Set.range ⇑P.coroot) α
    ⊢ Membership.mem (Set.range ⇑P.coroot) (HSMul.hSMul r α)
  -/
  obtain ⟨i, rfl⟩ := hr
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    α : N
    hα : Membership.mem (Set.range ⇑P.coroot) α
    i : ι
    ⊢ Membership.mem (Set.range ⇑P.coroot) (HSMul.hSMul (P.coreflection i) α)
  -/
  obtain ⟨j, rfl⟩ := hα
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Membership.mem (Set.range ⇑P.coroot) (HSMul.hSMul (P.coreflection i) (P.coro …
  -/
  exact ⟨P.reflection_perm i j, P.coroot_reflection_perm i j⟩
  /-
    🎉 no goals
  -/


lemma pairing_smul_root_eq (k : ι) (hij : P.reflection_perm i = P.reflection_perm j) :
    P.pairing k i • P.root i = P.pairing k j • P.root j := by
  have h : P.reflection i (P.root k) = P.reflection j (P.root k) := by
    simp only [← root_reflection_perm, hij]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j k : ι
    hij : Eq (P.reflection_perm i) (P.reflection_perm j)
    h : Eq ((P.reflection i) (P.root k)) ((P.reflection j) (P.root k))
    ⊢ Eq (HSMul.hSMul (P.pairing k i) (P.root i)) (HSMul.hSMul (P.pairing k j) (P. …
  -/
  simpa only [reflection_apply_root, sub_right_inj] using h
  /-
    🎉 no goals
  -/


lemma pairing_smul_coroot_eq (k : ι) (hij : P.reflection_perm i = P.reflection_perm j) :
    P.pairing i k • P.coroot i = P.pairing j k • P.coroot j := by
  have h : P.coreflection i (P.coroot k) = P.coreflection j (P.coroot k) := by
    simp only [← coroot_reflection_perm, hij]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j k : ι
    hij : Eq (P.reflection_perm i) (P.reflection_perm j)
    h : Eq ((P.coreflection i) (P.coroot k)) ((P.coreflection j) (P.coroot k))
    ⊢ Eq (HSMul.hSMul (P.pairing i k) (P.coroot i)) (HSMul.hSMul (P.pairing j k) ( …
  -/
  simpa only [coreflection_apply_coroot, sub_right_inj] using h
  /-
    🎉 no goals
  -/


lemma two_nsmul_reflection_eq_of_perm_eq (hij : P.reflection_perm i = P.reflection_perm j) :
    2 • ⇑(P.reflection i) = 2 • P.reflection j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    hij : Eq (P.reflection_perm i) (P.reflection_perm j)
    ⊢ Eq (HSMul.hSMul 2 ⇑(P.reflection i)) (HSMul.hSMul 2 ⇑(P.reflection j))
  -/
  ext x
  suffices 2 • P.toLin x (P.coroot i) • P.root i = 2 • P.toLin x (P.coroot j) • P.root j by
    simpa [reflection_apply, smul_sub]
  calc 2 • P.toLin x (P.coroot i) • P.root i
      = P.toLin x (P.coroot i) • ((2 : R) • P.root i) := ?_
    _ = P.toLin x (P.coroot i) • (P.pairing i j • P.root j) := ?_
    _ = P.toLin x (P.pairing i j • P.coroot i) • (P.root j) := ?_
    _ = P.toLin x ((2 : R) • P.coroot j) • (P.root j) := ?_
    _ = 2 • P.toLin x (P.coroot j) • P.root j := ?_
    /-
      case h.calc_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      hij : Eq (P.reflection_perm i) (P.reflection_perm j)
      x : M
      ⊢ Eq (HSMul.hSMul 2 (HSMul.hSMul ((P.toLin x) (P.coroot i)) (P.root i))) (HSMu …
    -/
  · rw [smul_comm, ← Nat.cast_smul_eq_nsmul R, Nat.cast_ofNat]
    /-
      🎉 no goals
    -/
    /-
      case h.calc_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      hij : Eq (P.reflection_perm i) (P.reflection_perm j)
      x : M
      ⊢ Eq (HSMul.hSMul ((P.toLin x) (P.coroot i)) (HSMul.hSMul 2 (P.root i))) (HSMu …
    -/
  · rw [P.pairing_smul_root_eq j i i hij.symm, pairing_same]
    /-
      🎉 no goals
    -/
    /-
      case h.calc_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      hij : Eq (P.reflection_perm i) (P.reflection_perm j)
      x : M
      ⊢ Eq (HSMul.hSMul ((P.toLin x) (P.coroot i)) (HSMul.hSMul (P.pairing i j) (P.r …
    -/
  · rw [← smul_comm, ← smul_assoc, map_smul]
    /-
      🎉 no goals
    -/
    /-
      case h.calc_4
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      hij : Eq (P.reflection_perm i) (P.reflection_perm j)
      x : M
      ⊢ Eq (HSMul.hSMul ((P.toLin x) (HSMul.hSMul (P.pairing i j) (P.coroot i))) (P. …
    -/
  · rw [← P.pairing_smul_coroot_eq j i j hij.symm, pairing_same]
    /-
      🎉 no goals
    -/
    /-
      case h.calc_5
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      hij : Eq (P.reflection_perm i) (P.reflection_perm j)
      x : M
      ⊢ Eq (HSMul.hSMul ((P.toLin x) (HSMul.hSMul 2 (P.coroot j))) (P.root j)) (HSMu …
    -/
  · rw [map_smul, smul_assoc, ← Nat.cast_smul_eq_nsmul R, Nat.cast_ofNat]
    /-
      🎉 no goals
    -/


lemma reflection_perm_eq_reflection_perm_iff_of_isSMulRegular (h2 : IsSMulRegular M 2) :
    P.reflection_perm i = P.reflection_perm j ↔ P.reflection i = P.reflection j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h2 : IsSMulRegular M 2
    ⊢ Iff (Eq (P.reflection_perm i) (P.reflection_perm j)) (Eq (P.reflection i) (P …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ Equiv.ext fun k ↦ P.root.injective <| by simp [h]⟩
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h2 : IsSMulRegular M 2
    h : Eq (P.reflection_perm i) (P.reflection_perm j)
    ⊢ Eq (P.reflection i) (P.reflection j)
  -/
  suffices ⇑(P.reflection i) = ⇑(P.reflection j) from DFunLike.coe_injective this
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h2 : IsSMulRegular M 2
    h : Eq (P.reflection_perm i) (P.reflection_perm j)
    ⊢ Eq ⇑(P.reflection i) ⇑(P.reflection j)
  -/
  replace h2 : IsSMulRegular (M → M) 2 := IsSMulRegular.pi fun _ ↦ h2
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h : Eq (P.reflection_perm i) (P.reflection_perm j)
    h2 : IsSMulRegular (M → M) 2
    ⊢ Eq ⇑(P.reflection i) ⇑(P.reflection j)
  -/
  exact h2 <| P.two_nsmul_reflection_eq_of_perm_eq i j h
  /-
    🎉 no goals
  -/


lemma reflection_perm_eq_reflection_perm_iff_of_span :
    P.reflection_perm i = P.reflection_perm j ↔
    ∀ x ∈ span R (range P.root), P.reflection i x = P.reflection j x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Iff (Eq (P.reflection_perm i) (P.reflection_perm j)) (∀ (x : M), Membership. …
  -/
  refine ⟨fun h x hx ↦ ?_, fun h ↦ ?_⟩
  · induction hx using Submodule.span_induction with
    | mem x hx =>
      obtain ⟨k, rfl⟩ := hx
      simp only [← root_reflection_perm, h]
    | zero => simp
    | add x y _ _ hx hy => simp [hx, hy]
    | smul t x _ hx => simp [hx]
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      h : ∀ (x : M), Membership.mem (Submodule.span R (Set.range ⇑P.root)) x → Eq (( …
      ⊢ Eq (P.reflection_perm i) (P.reflection_perm j)
    -/
  · ext k
    /-
      case refine_2.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      h : ∀ (x : M), Membership.mem (Submodule.span R (Set.range ⇑P.root)) x → Eq (( …
      k : ι
      ⊢ Eq ((P.reflection_perm i) k) ((P.reflection_perm j) k)
    -/
    apply P.root.injective
    /-
      case refine_2.H.a
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i j : ι
      h : ∀ (x : M), Membership.mem (Submodule.span R (Set.range ⇑P.root)) x → Eq (( …
      k : ι
      ⊢ Eq (P.root ((P.reflection_perm i) k)) (P.root ((P.reflection_perm j) k))
    -/
    simp [h (P.root k) (Submodule.subset_span <| mem_range_self k)]
    /-
      🎉 no goals
    -/


lemma _root_.RootSystem.reflection_perm_eq_reflection_perm_iff (P : RootSystem ι R M N) (i j : ι) :
    P.reflection_perm i = P.reflection_perm j ↔ P.reflection i = P.reflection j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootSystem ι R M N
    i j : ι
    ⊢ Iff (Eq (P.reflection_perm i) (P.reflection_perm j)) (Eq (P.reflection i) (P …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ Equiv.ext fun k ↦ P.root.injective <| by simp [h]⟩
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootSystem ι R M N
    i j : ι
    h : Eq (P.reflection_perm i) (P.reflection_perm j)
    ⊢ Eq (P.reflection i) (P.reflection j)
  -/
  ext x
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootSystem ι R M N
    i j : ι
    h : Eq (P.reflection_perm i) (P.reflection_perm j)
    x : M
    ⊢ Eq ((P.reflection i) x) ((P.reflection j) x)
  -/
  exact (P.reflection_perm_eq_reflection_perm_iff_of_span i j).mp h x <| by simp
  /-
    🎉 no goals
  -/


/-- The Coxeter Weight of a pair gives the weight of an edge in a Coxeter diagram, when it is
finite.  It is `4 cos² θ`, where `θ` describes the dihedral angle between hyperplanes. -/
def coxeterWeight : R := pairing P i j * pairing P j i


lemma coxeterWeight_swap : coxeterWeight P i j = coxeterWeight P j i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Eq (P.coxeterWeight i j) (P.coxeterWeight j i)
  -/
  simp only [coxeterWeight, mul_comm]
  /-
    🎉 no goals
  -/


lemma exists_int_eq_coxeterWeight [P.IsCrystallographic] (i j : ι) :
    ∃ z : ℤ, P.coxeterWeight i j = z := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsCrystallographic
    i j : ι
    ⊢ Exists fun z => Eq (P.coxeterWeight i j) ↑z
  -/
  obtain ⟨a, ha⟩ := P.exists_int i j
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsCrystallographic
    i j : ι
    a : Int
    ha : Eq (↑a) (P.pairing i j)
    ⊢ Exists fun z => Eq (P.coxeterWeight i j) ↑z
  -/
  obtain ⟨b, hb⟩ := P.exists_int j i
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsCrystallographic
    i j : ι
    a : Int
    ha : Eq (↑a) (P.pairing i j)
    b : Int
    hb : Eq (↑b) (P.pairing j i)
    ⊢ Exists fun z => Eq (P.coxeterWeight i j) ↑z
  -/
  exact ⟨a * b, by simp [coxeterWeight, ha, hb]⟩
  /-
    🎉 no goals
  -/


/-- Two roots are orthogonal when they are fixed by each others' reflections. -/
def IsOrthogonal : Prop := pairing P i j = 0 ∧ pairing P j i = 0


lemma isOrthogonal_symm : IsOrthogonal P i j ↔ IsOrthogonal P j i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ Iff (P.IsOrthogonal i j) (P.IsOrthogonal j i)
  -/
  simp only [IsOrthogonal, and_comm]
  /-
    🎉 no goals
  -/


lemma IsOrthogonal.symm (h : IsOrthogonal P i j) : IsOrthogonal P j i :=
  ⟨h.2, h.1⟩


lemma isOrthogonal_comm (h : IsOrthogonal P i j) : Commute (P.reflection i) (P.reflection j) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h : P.IsOrthogonal i j
    ⊢ Commute (P.reflection i) (P.reflection j)
  -/
  rw [commute_iff_eq]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h : P.IsOrthogonal i j
    ⊢ Eq (HMul.hMul (P.reflection i) (P.reflection j)) (HMul.hMul (P.reflection j) …
  -/
  ext v
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    h : P.IsOrthogonal i j
    v : M
    ⊢ Eq ((HMul.hMul (P.reflection i) (P.reflection j)) v) ((HMul.hMul (P.reflecti …
  -/
  replace h : P.pairing i j = 0 ∧ P.pairing j i = 0 := by simpa [IsOrthogonal] using h
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    v : M
    h : And (Eq (P.pairing i j) 0) (Eq (P.pairing j i) 0)
    ⊢ Eq ((HMul.hMul (P.reflection i) (P.reflection j)) v) ((HMul.hMul (P.reflecti …
  -/
  erw [LinearMap.mul_apply, LinearMap.mul_apply]
  simp only [LinearEquiv.coe_coe, reflection_apply, PerfectPairing.flip_apply_apply, map_sub,
    map_smul, root_coroot_eq_pairing, h, zero_smul, sub_zero]
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    v : M
    h : And (Eq (P.pairing i j) 0) (Eq (P.pairing j i) 0)
    ⊢ Eq (HSub.hSub (HSub.hSub v (HSMul.hSMul ((P.toPerfectPairing v) (P.coroot i) …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


