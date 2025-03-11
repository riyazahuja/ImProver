/-- Type synonym for turning a `LieRing` into a `NonUnitalNonAssocRing`.

A `LieRing` can be regarded as a `NonUnitalNonAssocRing` by turning its
`Bracket` (denoted `⁅, ⁆`) into a `Mul` (denoted `*`). -/
def CommutatorRing (L : Type v) : Type v := L


/-- A `LieRing` can be regarded as a `NonUnitalNonAssocRing` by turning its
`Bracket` (denoted `⁅, ⁆`) into a `Mul` (denoted `*`). -/
instance : NonUnitalNonAssocRing (CommutatorRing L) :=
  show NonUnitalNonAssocRing L from
    { (inferInstance : AddCommGroup L) with
      mul := Bracket.bracket
      left_distrib := lie_add
      right_distrib := add_lie
      zero_mul := zero_lie
      mul_zero := lie_zero }


instance (L : Type v) [Nonempty L] : Nonempty (CommutatorRing L) := ‹Nonempty L›


instance (L : Type v) [Inhabited L] : Inhabited (CommutatorRing L) := ‹Inhabited L›


                                                           /-
                                                             R : Type u
                                                             L : Type v
                                                             inst✝² : CommRing R
                                                             inst✝¹ : LieRing L
                                                             inst✝ : LieAlgebra R L
                                                             ⊢ LieRing L
                                                           -/
instance : LieRing (CommutatorRing L) := show LieRing L by infer_instance
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                                     /-
                                                                       R : Type u
                                                                       L : Type v
                                                                       inst✝² : CommRing R
                                                                       inst✝¹ : LieRing L
                                                                       inst✝ : LieAlgebra R L
                                                                       ⊢ LieAlgebra R L
                                                                     -/
instance : LieAlgebra R (CommutatorRing L) := show LieAlgebra R L by infer_instance
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- Regarding the `LieRing` of a `LieAlgebra` as a `NonUnitalNonAssocRing`, we can
reinterpret the `smul_lie` law as an `IsScalarTower`. -/
instance isScalarTower : IsScalarTower R (CommutatorRing L) (CommutatorRing L) := ⟨smul_lie⟩


/-- Regarding the `LieRing` of a `LieAlgebra` as a `NonUnitalNonAssocRing`, we can
reinterpret the `lie_smul` law as an `SMulCommClass`. -/
instance smulCommClass : SMulCommClass R (CommutatorRing L) (CommutatorRing L) :=
  ⟨fun t x y => (lie_smul t x y).symm⟩


/-- Regarding the `LieRing` of a `LieAlgebra` as a `NonUnitalNonAssocRing`, we can
regard a `LieHom` as a `NonUnitalAlgHom`. -/
@[simps]
def toNonUnitalAlgHom (f : L →ₗ⁅R⁆ L₂) : CommutatorRing L →ₙₐ[R] CommutatorRing L₂ :=
  { f with
    toFun := f
    map_zero' := f.map_zero
    map_mul' := f.map_lie }


theorem toNonUnitalAlgHom_injective :
    Function.Injective (toNonUnitalAlgHom : _ → CommutatorRing L →ₙₐ[R] CommutatorRing L₂) :=
  fun _ _ h => ext <| NonUnitalAlgHom.congr_fun h


