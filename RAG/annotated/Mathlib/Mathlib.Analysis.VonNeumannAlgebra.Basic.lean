/-- Sakai's definition of a von Neumann algebra as a C^* algebra with a Banach space predual.

So that we can unambiguously talk about these "abstract" von Neumann algebras
in parallel with the "concrete" ones (weakly closed *-subalgebras of B(H)),
we name this definition `WStarAlgebra`.

Note that for now we only assert the mere existence of predual, rather than picking one.
This may later prove problematic, and need to be revisited.
Picking one may cause problems with definitional unification of different instances.
One the other hand, not picking one means that the weak-* topology
(which depends on a choice of predual) must be defined using the choice,
and we may be unhappy with the resulting opaqueness of the definition.
-/
class WStarAlgebra (M : Type u) [CStarAlgebra M] : Prop where
  /-- There is a Banach space `X` whose dual is isometrically (conjugate-linearly) isomorphic
  to the `WStarAlgebra`. -/
  exists_predual :
    ∃ (X : Type u) (_ : NormedAddCommGroup X) (_ : NormedSpace ℂ X) (_ : CompleteSpace X),
      Nonempty (NormedSpace.Dual ℂ X ≃ₗᵢ⋆[ℂ] M)

-- TODO: Without this, `VonNeumannAlgebra` times out. Why?

/-- The double commutant definition of a von Neumann algebra,
as a *-closed subalgebra of bounded operators on a Hilbert space,
which is equal to its double commutant.

Note that this definition is parameterised by the Hilbert space
on which the algebra faithfully acts, as is standard in the literature.
See `WStarAlgebra` for the abstract notion (a C^*-algebra with Banach space predual).

Note this is a bundled structure, parameterised by the Hilbert space `H`,
rather than a typeclass on the type of elements.
Thus we can't say that the bounded operators `H →L[ℂ] H` form a `VonNeumannAlgebra`
(although we will later construct the instance `WStarAlgebra (H →L[ℂ] H)`),
and instead will use `⊤ : VonNeumannAlgebra H`.
-/
-- Porting note: I don't think the nonempty instance linter exists yet
structure VonNeumannAlgebra (H : Type u) [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H] extends StarSubalgebra ℂ (H →L[ℂ] H) where
  /-- The double commutant (a.k.a. centralizer) of a `VonNeumannAlgebra` is itself. -/
  centralizer_centralizer' : Set.centralizer (Set.centralizer carrier) = carrier


instance instSetLike : SetLike (VonNeumannAlgebra H) (H →L[ℂ] H) where
  coe S := S.carrier
                             /-
                               H : Type u
                               inst✝² : NormedAddCommGroup H
                               inst✝¹ : InnerProductSpace Complex H
                               inst✝ : CompleteSpace H
                               S T : VonNeumannAlgebra H
                               h : Eq ((fun S => S.carrier) S) ((fun S => S.carrier) T)
                               ⊢ Eq S T
                             -/
  coe_injective' S T h := by obtain ⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, _⟩, _⟩, _⟩ := S; cases T; congr
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/

-- Porting note: `StarMemClass` should be in `Prop`?

noncomputable instance instStarMemClass : StarMemClass (VonNeumannAlgebra H) (H →L[ℂ] H) where
  star_mem {s} := s.star_mem'


instance instSubringClass : SubringClass (VonNeumannAlgebra H) (H →L[ℂ] H) where
  add_mem {s} := s.add_mem'
  mul_mem {s} := s.mul_mem'
  one_mem {s} := s.one_mem'
  zero_mem {s} := s.zero_mem'
  neg_mem {s} a ha := show -a ∈ s.toStarSubalgebra from neg_mem ha


@[simp]
theorem mem_carrier {S : VonNeumannAlgebra H} {x : H →L[ℂ] H} :
    x ∈ S.toStarSubalgebra ↔ x ∈ (S : Set (H →L[ℂ] H)) :=
  Iff.rfl
-- Porting note: changed the declaration because `simpNF` indicated the LHS simplifies to this.


@[simp]
theorem coe_toStarSubalgebra (S : VonNeumannAlgebra H) :
    (S.toStarSubalgebra : Set (H →L[ℂ] H)) = S :=
  rfl


@[simp]
theorem coe_mk (S : StarSubalgebra ℂ (H →L[ℂ] H)) (h) :
    ((⟨S, h⟩ : VonNeumannAlgebra H) : Set (H →L[ℂ] H)) = S :=
  rfl


@[ext]
theorem ext {S T : VonNeumannAlgebra H} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[simp]
theorem centralizer_centralizer (S : VonNeumannAlgebra H) :
    Set.centralizer (Set.centralizer (S : Set (H →L[ℂ] H))) = S :=
  S.centralizer_centralizer'


/-- The centralizer of a `VonNeumannAlgebra`, as a `VonNeumannAlgebra`. -/
def commutant (S : VonNeumannAlgebra H) : VonNeumannAlgebra H where
  toStarSubalgebra := StarSubalgebra.centralizer ℂ (S : Set (H →L[ℂ] H))
                                 /-
                                   H : Type u
                                   inst✝² : NormedAddCommGroup H
                                   inst✝¹ : InnerProductSpace Complex H
                                   inst✝ : CompleteSpace H
                                   S : VonNeumannAlgebra H
                                   ⊢ Eq (StarSubalgebra.centralizer Complex ↑S).carrier.centralizer.centralizer ( …
                                 -/
  centralizer_centralizer' := by simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem coe_commutant (S : VonNeumannAlgebra H) :
    ↑S.commutant = Set.centralizer (S : Set (H →L[ℂ] H)) := by
  /-
    H : Type u
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Complex H
    inst✝ : CompleteSpace H
    S : VonNeumannAlgebra H
    ⊢ Eq (↑S.commutant) (↑S).centralizer
  -/
  simp [commutant]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_commutant_iff {S : VonNeumannAlgebra H} {z : H →L[ℂ] H} :
    z ∈ S.commutant ↔ ∀ g ∈ S, g * z = z * g := by
  /-
    H : Type u
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Complex H
    inst✝ : CompleteSpace H
    S : VonNeumannAlgebra H
    z : ContinuousLinearMap (RingHom.id Complex) H H
    ⊢ Iff (Membership.mem S.commutant z) (∀ (g : ContinuousLinearMap (RingHom.id C …
  -/
  rw [← SetLike.mem_coe, coe_commutant]
  /-
    H : Type u
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Complex H
    inst✝ : CompleteSpace H
    S : VonNeumannAlgebra H
    z : ContinuousLinearMap (RingHom.id Complex) H H
    ⊢ Iff (Membership.mem (↑S).centralizer z) (∀ (g : ContinuousLinearMap (RingHom …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem commutant_commutant (S : VonNeumannAlgebra H) : S.commutant.commutant = S :=
                              /-
                                H : Type u
                                inst✝² : NormedAddCommGroup H
                                inst✝¹ : InnerProductSpace Complex H
                                inst✝ : CompleteSpace H
                                S : VonNeumannAlgebra H
                                ⊢ Eq ↑S.commutant.commutant ↑S
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


