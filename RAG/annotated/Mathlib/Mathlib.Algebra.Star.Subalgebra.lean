/-- A *-subalgebra is a subalgebra of a *-algebra which is closed under *. -/
structure StarSubalgebra (R : Type u) (A : Type v) [CommSemiring R] [StarRing R] [Semiring A]
  [StarRing A] [Algebra R A] [StarModule R A] extends Subalgebra R A : Type v where
  /-- The `carrier` is closed under the `star` operation. -/
  star_mem' {a} : a ∈ carrier → star a ∈ carrier


instance setLike : SetLike (StarSubalgebra R A) A where
  coe S := S.carrier
                             /-
                               F : Type u_1
                               R : Type u_2
                               A : Type u_3
                               B : Type u_4
                               C : Type u_5
                               inst✝¹³ : CommSemiring R
                               inst✝¹² : StarRing R
                               inst✝¹¹ : Semiring A
                               inst✝¹⁰ : StarRing A
                               inst✝⁹ : Algebra R A
                               inst✝⁸ : StarModule R A
                               inst✝⁷ : Semiring B
                               inst✝⁶ : StarRing B
                               inst✝⁵ : Algebra R B
                               inst✝⁴ : StarModule R B
                               inst✝³ : Semiring C
                               inst✝² : StarRing C
                               inst✝¹ : Algebra R C
                               inst✝ : StarModule R C
                               p q : StarSubalgebra R A
                               h : Eq ((fun S => S.carrier) p) ((fun S => S.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by obtain ⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, _⟩, _⟩ := p; cases q; congr
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance starMemClass : StarMemClass (StarSubalgebra R A) A where
  star_mem {s} := s.star_mem'



instance subsemiringClass : SubsemiringClass (StarSubalgebra R A) A where
  add_mem {s} := s.add_mem'
  mul_mem {s} := s.mul_mem'
  one_mem {s} := s.one_mem'
  zero_mem {s} := s.zero_mem'


instance smulMemClass : SMulMemClass (StarSubalgebra R A) R A where
  smul_mem {s} r a (ha : a ∈ s.toSubalgebra) :=
    (SMulMemClass.smul_mem r ha : r • a ∈ s.toSubalgebra)


instance subringClass {R A} [CommRing R] [StarRing R] [Ring A] [StarRing A] [Algebra R A]
    [StarModule R A] : SubringClass (StarSubalgebra R A) A where
  neg_mem {s a} ha := show -a ∈ s.toSubalgebra from neg_mem ha

-- this uses the `Star` instance `s` inherits from `StarMemClass (StarSubalgebra R A) A`

instance starRing (s : StarSubalgebra R A) : StarRing s :=
  { StarMemClass.instStar s with
    star_involutive := fun r => Subtype.ext (star_star (r : A))
    star_mul := fun r₁ r₂ => Subtype.ext (star_mul (r₁ : A) (r₂ : A))
    star_add := fun r₁ r₂ => Subtype.ext (star_add (r₁ : A) (r₂ : A)) }


instance algebra (s : StarSubalgebra R A) : Algebra R s :=
  s.toSubalgebra.algebra'


instance starModule (s : StarSubalgebra R A) : StarModule R s where
  star_smul r a := Subtype.ext (star_smul r (a : A))


theorem mem_carrier {s : StarSubalgebra R A} {x : A} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


@[ext]
theorem ext {S T : StarSubalgebra R A} (h : ∀ x : A, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[simp]
lemma coe_mk (S : Subalgebra R A) (h) : ((⟨S, h⟩ : StarSubalgebra R A) : Set A) = S := rfl


@[simp]
theorem mem_toSubalgebra {S : StarSubalgebra R A} {x} : x ∈ S.toSubalgebra ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_toSubalgebra (S : StarSubalgebra R A) : (S.toSubalgebra : Set A) = S :=
  rfl


theorem toSubalgebra_injective :
    Function.Injective (toSubalgebra : StarSubalgebra R A → Subalgebra R A) := fun S T h =>
                  /-
                    R : Type u_2
                    A : Type u_3
                    inst✝⁵ : CommSemiring R
                    inst✝⁴ : StarRing R
                    inst✝³ : Semiring A
                    inst✝² : StarRing A
                    inst✝¹ : Algebra R A
                    inst✝ : StarModule R A
                    S T : StarSubalgebra R A
                    h : Eq S.toSubalgebra T.toSubalgebra
                    x : A
                    ⊢ Iff (Membership.mem S x) (Membership.mem T x)
                  -/
  ext fun x => by rw [← mem_toSubalgebra, ← mem_toSubalgebra, h]
                  /-
                    🎉 no goals
                  -/


theorem toSubalgebra_inj {S U : StarSubalgebra R A} : S.toSubalgebra = U.toSubalgebra ↔ S = U :=
  toSubalgebra_injective.eq_iff


theorem toSubalgebra_le_iff {S₁ S₂ : StarSubalgebra R A} :
    S₁.toSubalgebra ≤ S₂.toSubalgebra ↔ S₁ ≤ S₂ :=
  Iff.rfl


/-- Copy of a star subalgebra with a new `carrier` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (S : StarSubalgebra R A) (s : Set A) (hs : s = ↑S) : StarSubalgebra R A where
  toSubalgebra := Subalgebra.copy S.toSubalgebra s hs
                                           /-
                                             F : Type u_1
                                             R : Type u_2
                                             A : Type u_3
                                             B : Type u_4
                                             C : Type u_5
                                             inst✝¹³ : CommSemiring R
                                             inst✝¹² : StarRing R
                                             inst✝¹¹ : Semiring A
                                             inst✝¹⁰ : StarRing A
                                             inst✝⁹ : Algebra R A
                                             inst✝⁸ : StarModule R A
                                             inst✝⁷ : Semiring B
                                             inst✝⁶ : StarRing B
                                             inst✝⁵ : Algebra R B
                                             inst✝⁴ : StarModule R B
                                             inst✝³ : Semiring C
                                             inst✝² : StarRing C
                                             inst✝¹ : Algebra R C
                                             inst✝ : StarModule R C
                                             S : StarSubalgebra R A
                                             s : Set A
                                             hs : Eq s ↑S
                                             a : A
                                             ha : Membership.mem (S.copy s hs).carrier a
                                             ⊢ Membership.mem S.carrier a
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  star_mem' {a} ha := hs ▸ S.star_mem' (by simpa [hs] using ha)
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem coe_copy (S : StarSubalgebra R A) (s : Set A) (hs : s = ↑S) : (S.copy s hs : Set A) = s :=
  rfl


theorem copy_eq (S : StarSubalgebra R A) (s : Set A) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


theorem algebraMap_mem (r : R) : algebraMap R A r ∈ S :=
  S.algebraMap_mem' r


theorem rangeS_le : (algebraMap R A).rangeS ≤ S.toSubalgebra.toSubsemiring := fun _x ⟨r, hr⟩ =>
  hr ▸ S.algebraMap_mem r


theorem range_subset : Set.range (algebraMap R A) ⊆ S := fun _x ⟨r, hr⟩ => hr ▸ S.algebraMap_mem r


theorem range_le : Set.range (algebraMap R A) ≤ S :=
  S.range_subset


protected theorem smul_mem {x : A} (hx : x ∈ S) (r : R) : r • x ∈ S :=
  (Algebra.smul_def r x).symm ▸ mul_mem (S.algebraMap_mem r) hx


/-- Embedding of a subalgebra into the algebra. -/
def subtype : S →⋆ₐ[R] A where
  toFun := ((↑) : S → A)
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl
  commutes' _ := rfl
  map_star' _ := rfl


@[simp]
theorem coe_subtype : (S.subtype : S → A) = Subtype.val :=
  rfl


theorem subtype_apply (x : S) : S.subtype x = (x : A) :=
  rfl


@[simp]
theorem toSubalgebra_subtype : S.toSubalgebra.val = S.subtype.toAlgHom :=
  rfl


/-- The inclusion map between `StarSubalgebra`s given by `Subtype.map id` as a `StarAlgHom`. -/
@[simps]
def inclusion {S₁ S₂ : StarSubalgebra R A} (h : S₁ ≤ S₂) : S₁ →⋆ₐ[R] S₂ where
  toFun := Subtype.map id h
  map_one' := rfl
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl
  commutes' _ := rfl
  map_star' _ := rfl


theorem inclusion_injective {S₁ S₂ : StarSubalgebra R A} (h : S₁ ≤ S₂) :
    Function.Injective <| inclusion h :=
  Set.inclusion_injective h


@[simp]
theorem subtype_comp_inclusion {S₁ S₂ : StarSubalgebra R A} (h : S₁ ≤ S₂) :
    S₂.subtype.comp (inclusion h) = S₁.subtype :=
  rfl


/-- Transport a star subalgebra via a star algebra homomorphism. -/
def map (f : A →⋆ₐ[R] B) (S : StarSubalgebra R A) : StarSubalgebra R B :=
  { S.toSubalgebra.map f.toAlgHom with
    star_mem' := by
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        C : Type u_5
        inst✝¹³ : CommSemiring R
        inst✝¹² : StarRing R
        inst✝¹¹ : Semiring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra R A
        inst✝⁸ : StarModule R A
        inst✝⁷ : Semiring B
        inst✝⁶ : StarRing B
        inst✝⁵ : Algebra R B
        inst✝⁴ : StarModule R B
        inst✝³ : Semiring C
        inst✝² : StarRing C
        inst✝¹ : Algebra R C
        inst✝ : StarModule R C
        S✝ : StarSubalgebra R A
        f : StarAlgHom R A B
        S : StarSubalgebra R A
        ⊢ ∀ {a : B}, Membership.mem __src✝.carrier a → Membership.mem __src✝.carrier ( …
      -/
      rintro _ ⟨a, ha, rfl⟩
      /-
        case intro.intro
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        C : Type u_5
        inst✝¹³ : CommSemiring R
        inst✝¹² : StarRing R
        inst✝¹¹ : Semiring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra R A
        inst✝⁸ : StarModule R A
        inst✝⁷ : Semiring B
        inst✝⁶ : StarRing B
        inst✝⁵ : Algebra R B
        inst✝⁴ : StarModule R B
        inst✝³ : Semiring C
        inst✝² : StarRing C
        inst✝¹ : Algebra R C
        inst✝ : StarModule R C
        S✝ : StarSubalgebra R A
        f : StarAlgHom R A B
        S : StarSubalgebra R A
        a : A
        ha : Membership.mem (↑S.toSubsemiring) a
        ⊢ Membership.mem __src✝.carrier (Star.star (↑f.toAlgHom a))
      -/
      exact map_star f a ▸ Set.mem_image_of_mem _ (S.star_mem' ha) }
      /-
        🎉 no goals
      -/


theorem map_mono {S₁ S₂ : StarSubalgebra R A} {f : A →⋆ₐ[R] B} : S₁ ≤ S₂ → S₁.map f ≤ S₂.map f :=
  Set.image_subset f


theorem map_injective {f : A →⋆ₐ[R] B} (hf : Function.Injective f) : Function.Injective (map f) :=
  fun _S₁ _S₂ ih =>
  ext <| Set.ext_iff.1 <| Set.image_injective.2 hf <| Set.ext <| SetLike.ext_iff.mp ih


@[simp]
theorem map_id (S : StarSubalgebra R A) : S.map (StarAlgHom.id R A) = S :=
  SetLike.coe_injective <| Set.image_id _


theorem map_map (S : StarSubalgebra R A) (g : B →⋆ₐ[R] C) (f : A →⋆ₐ[R] B) :
    (S.map f).map g = S.map (g.comp f) :=
  SetLike.coe_injective <| Set.image_image _ _ _


@[simp]
theorem mem_map {S : StarSubalgebra R A} {f : A →⋆ₐ[R] B} {y : B} :
    y ∈ map f S ↔ ∃ x ∈ S, f x = y :=
  Subsemiring.mem_map


theorem map_toSubalgebra {S : StarSubalgebra R A} {f : A →⋆ₐ[R] B} :
    (S.map f).toSubalgebra = S.toSubalgebra.map f.toAlgHom :=
  SetLike.coe_injective rfl


@[simp]
theorem coe_map (S : StarSubalgebra R A) (f : A →⋆ₐ[R] B) : (S.map f : Set B) = f '' S :=
  rfl


/-- Preimage of a star subalgebra under a star algebra homomorphism. -/
def comap (f : A →⋆ₐ[R] B) (S : StarSubalgebra R B) : StarSubalgebra R A :=
  { S.toSubalgebra.comap f.toAlgHom with
    star_mem' := @fun a ha => show f (star a) ∈ S from (map_star f a).symm ▸ star_mem ha }


theorem map_le_iff_le_comap {S : StarSubalgebra R A} {f : A →⋆ₐ[R] B} {U : StarSubalgebra R B} :
    map f S ≤ U ↔ S ≤ comap f U :=
  Set.image_subset_iff


theorem gc_map_comap (f : A →⋆ₐ[R] B) : GaloisConnection (map f) (comap f) := fun _S _U =>
  map_le_iff_le_comap


theorem comap_mono {S₁ S₂ : StarSubalgebra R B} {f : A →⋆ₐ[R] B} :
    S₁ ≤ S₂ → S₁.comap f ≤ S₂.comap f :=
  Set.preimage_mono


theorem comap_injective {f : A →⋆ₐ[R] B} (hf : Function.Surjective f) :
    Function.Injective (comap f) := fun _S₁ _S₂ h =>
  ext fun b =>
    let ⟨x, hx⟩ := hf b
    let this := SetLike.ext_iff.1 h x
    hx ▸ this


@[simp]
theorem comap_id (S : StarSubalgebra R A) : S.comap (StarAlgHom.id R A) = S :=
  SetLike.coe_injective <| Set.preimage_id


theorem comap_comap (S : StarSubalgebra R C) (g : B →⋆ₐ[R] C) (f : A →⋆ₐ[R] B) :
    (S.comap g).comap f = S.comap (g.comp f) :=
                              /-
                                R : Type u_2
                                A : Type u_3
                                B : Type u_4
                                C : Type u_5
                                inst✝¹³ : CommSemiring R
                                inst✝¹² : StarRing R
                                inst✝¹¹ : Semiring A
                                inst✝¹⁰ : StarRing A
                                inst✝⁹ : Algebra R A
                                inst✝⁸ : StarModule R A
                                inst✝⁷ : Semiring B
                                inst✝⁶ : StarRing B
                                inst✝⁵ : Algebra R B
                                inst✝⁴ : StarModule R B
                                inst✝³ : Semiring C
                                inst✝² : StarRing C
                                inst✝¹ : Algebra R C
                                inst✝ : StarModule R C
                                S : StarSubalgebra R C
                                g : StarAlgHom R B C
                                f : StarAlgHom R A B
                                ⊢ Eq ↑(StarSubalgebra.comap f (StarSubalgebra.comap g S)) ↑(StarSubalgebra.com …
                              -/
  SetLike.coe_injective <| by exact Set.preimage_preimage
                              /-
                                🎉 no goals
                              -/
  -- Porting note: the `by exact` trick still works sometimes


@[simp]
theorem mem_comap (S : StarSubalgebra R B) (f : A →⋆ₐ[R] B) (x : A) : x ∈ S.comap f ↔ f x ∈ S :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_comap (S : StarSubalgebra R B) (f : A →⋆ₐ[R] B) :
    (S.comap f : Set A) = f ⁻¹' (S : Set B) :=
  rfl


/-- The centralizer, or commutant, of the star-closure of a set as a star subalgebra. -/
def centralizer (s : Set A) : StarSubalgebra R A where
  toSubalgebra := Subalgebra.centralizer R (s ∪ star s)
  star_mem' := Set.star_mem_centralizer


@[simp, norm_cast]
theorem coe_centralizer (s : Set A) : (centralizer R s : Set A) = (s ∪ star s).centralizer :=
  rfl


open Set in
nonrec theorem mem_centralizer_iff {s : Set A} {z : A} :
    z ∈ centralizer R s ↔ ∀ g ∈ s, g * z = z * g ∧ star g * z = z * star g := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    s : Set A
    z : A
    ⊢ Iff (Membership.mem (StarSubalgebra.centralizer R s) z) (∀ (g : A), Membersh …
  -/
  simp [← SetLike.mem_coe, centralizer_union, ← image_star, mem_centralizer_iff, forall_and]
  /-
    🎉 no goals
  -/


theorem centralizer_le (s t : Set A) (h : s ⊆ t) : centralizer R t ≤ centralizer R s :=
  Set.centralizer_subset (Set.union_subset_union h <| Set.preimage_mono h)


theorem centralizer_toSubalgebra (s : Set A) :
    (centralizer R s).toSubalgebra = Subalgebra.centralizer R (s ∪ star s):=
  rfl


theorem coe_centralizer_centralizer (s : Set A) :
    (centralizer R (centralizer R s : Set A)) = (s ∪ star s).centralizer.centralizer := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (↑(StarSubalgebra.centralizer R ↑(StarSubalgebra.centralizer R s))) (Unio …
  -/
  rw [coe_centralizer, StarMemClass.star_coe_eq, Set.union_self, coe_centralizer]
  /-
    🎉 no goals
  -/


/-- The pointwise `star` of a subalgebra is a subalgebra. -/
instance involutiveStar : InvolutiveStar (Subalgebra R A) where
  star S :=
    { carrier := star S.carrier
      mul_mem' := fun {x y} hx hy => by
        /-
          F : Type u_1
          R : Type u_2
          A : Type u_3
          B : Type u_4
          inst✝⁹ : CommSemiring R
          inst✝⁸ : StarRing R
          inst✝⁷ : Semiring A
          inst✝⁶ : Algebra R A
          inst✝⁵ : StarRing A
          inst✝⁴ : StarModule R A
          inst✝³ : Semiring B
          inst✝² : Algebra R B
          inst✝¹ : StarRing B
          inst✝ : StarModule R B
          S : Subalgebra R A
          x y : A
          hx : Membership.mem (Star.star S.carrier) x
          hy : Membership.mem (Star.star S.carrier) y
          ⊢ Membership.mem (Star.star S.carrier) (HMul.hMul x y)
        -/
        simp only [Set.mem_star, Subalgebra.mem_carrier] at *
        /-
          F : Type u_1
          R : Type u_2
          A : Type u_3
          B : Type u_4
          inst✝⁹ : CommSemiring R
          inst✝⁸ : StarRing R
          inst✝⁷ : Semiring A
          inst✝⁶ : Algebra R A
          inst✝⁵ : StarRing A
          inst✝⁴ : StarModule R A
          inst✝³ : Semiring B
          inst✝² : Algebra R B
          inst✝¹ : StarRing B
          inst✝ : StarModule R B
          S : Subalgebra R A
          x y : A
          hx : Membership.mem S (Star.star x)
          hy : Membership.mem S (Star.star y)
          ⊢ Membership.mem S (Star.star (HMul.hMul x y))
        -/
        exact (star_mul x y).symm ▸ mul_mem hy hx
        /-
          🎉 no goals
        -/
      one_mem' := Set.mem_star.mp ((star_one A).symm ▸ one_mem S : star (1 : A) ∈ S)
      add_mem' := fun {x y} hx hy => by
        /-
          F : Type u_1
          R : Type u_2
          A : Type u_3
          B : Type u_4
          inst✝⁹ : CommSemiring R
          inst✝⁸ : StarRing R
          inst✝⁷ : Semiring A
          inst✝⁶ : Algebra R A
          inst✝⁵ : StarRing A
          inst✝⁴ : StarModule R A
          inst✝³ : Semiring B
          inst✝² : Algebra R B
          inst✝¹ : StarRing B
          inst✝ : StarModule R B
          S : Subalgebra R A
          x y : A
          hx : Membership.mem { carrier := Star.star S.carrier, mul_mem' := ⋯, one_mem'  …
          hy : Membership.mem { carrier := Star.star S.carrier, mul_mem' := ⋯, one_mem'  …
          ⊢ Membership.mem { carrier := Star.star S.carrier, mul_mem' := ⋯, one_mem' :=  …
        -/
        simp only [Set.mem_star, Subalgebra.mem_carrier] at *
        /-
          F : Type u_1
          R : Type u_2
          A : Type u_3
          B : Type u_4
          inst✝⁹ : CommSemiring R
          inst✝⁸ : StarRing R
          inst✝⁷ : Semiring A
          inst✝⁶ : Algebra R A
          inst✝⁵ : StarRing A
          inst✝⁴ : StarModule R A
          inst✝³ : Semiring B
          inst✝² : Algebra R B
          inst✝¹ : StarRing B
          inst✝ : StarModule R B
          S : Subalgebra R A
          x y : A
          hx : Membership.mem S (Star.star x)
          hy : Membership.mem S (Star.star y)
          ⊢ Membership.mem S (Star.star (HAdd.hAdd x y))
        -/
        exact (star_add x y).symm ▸ add_mem hx hy
        /-
          🎉 no goals
        -/
      zero_mem' := Set.mem_star.mp ((star_zero A).symm ▸ zero_mem S : star (0 : A) ∈ S)
      algebraMap_mem' := fun r => by
        simpa only [Set.mem_star, Subalgebra.mem_carrier, ← algebraMap_star_comm] using
          S.algebraMap_mem (star r) }
  star_involutive S :=
    Subalgebra.ext fun x =>
      ⟨fun hx => star_star x ▸ hx, fun hx => ((star_star x).symm ▸ hx : star (star x) ∈ S)⟩


@[simp]
theorem mem_star_iff (S : Subalgebra R A) (x : A) : x ∈ star S ↔ star x ∈ S :=
  Iff.rfl

-- Porting note: removed `@[simp]` tag because `simp` can prove this

theorem star_mem_star_iff (S : Subalgebra R A) (x : A) : star x ∈ star S ↔ x ∈ S := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    S : Subalgebra R A
    x : A
    ⊢ Iff (Membership.mem (Star.star S) (Star.star x)) (Membership.mem S x)
  -/
  simp only [mem_star_iff, star_star]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_star (S : Subalgebra R A) : ((star S : Subalgebra R A) : Set A) = star (S : Set A) :=
  rfl


theorem star_mono : Monotone (star : Subalgebra R A → Subalgebra R A) := fun _ _ h _ hx => h hx


/-- The star operation on `Subalgebra` commutes with `Algebra.adjoin`. -/
theorem star_adjoin_comm (s : Set A) : star (Algebra.adjoin R s) = Algebra.adjoin R (star s) :=
  have this : ∀ t : Set A, Algebra.adjoin R (star t) ≤ star (Algebra.adjoin R t) := fun _ =>
    Algebra.adjoin_le fun _ hx => Algebra.subset_adjoin hx
                  /-
                    R : Type u_2
                    A : Type u_3
                    inst✝⁵ : CommSemiring R
                    inst✝⁴ : StarRing R
                    inst✝³ : Semiring A
                    inst✝² : Algebra R A
                    inst✝¹ : StarRing A
                    inst✝ : StarModule R A
                    s : Set A
                    this : ∀ (t : Set A), LE.le (Algebra.adjoin R (Star.star t)) (Star.star (Algeb …
                    ⊢ LE.le (Star.star (Algebra.adjoin R s)) (Algebra.adjoin R (Star.star s))
                  -/
  le_antisymm (by simpa only [star_star] using Subalgebra.star_mono (this (star s))) (this s)
                  /-
                    🎉 no goals
                  -/


/-- The `StarSubalgebra` obtained from `S : Subalgebra R A` by taking the smallest subalgebra
containing both `S` and `star S`. -/
@[simps!]
def starClosure (S : Subalgebra R A) : StarSubalgebra R A where
  toSubalgebra := S ⊔ star S
  star_mem' := fun {a} ha => by
    /-
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : StarRing A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring B
      inst✝² : Algebra R B
      inst✝¹ : StarRing B
      inst✝ : StarModule R B
      S : Subalgebra R A
      a : A
      ha : Membership.mem (Max.max S (Star.star S)).carrier a
      ⊢ Membership.mem (Max.max S (Star.star S)).carrier (Star.star a)
    -/
    simp only [Subalgebra.mem_carrier, ← (@Algebra.gi R A _ _ _).l_sup_u _ _] at *
    /-
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : StarRing A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring B
      inst✝² : Algebra R B
      inst✝¹ : StarRing B
      inst✝ : StarModule R B
      S : Subalgebra R A
      a : A
      ha : Membership.mem (Algebra.adjoin R (Max.max ↑S ↑(Star.star S))) a
      ⊢ Membership.mem (Algebra.adjoin R (Max.max ↑S ↑(Star.star S))) (Star.star a)
    -/
    rw [← mem_star_iff _ a, star_adjoin_comm, sup_comm]
    /-
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring R
      inst✝⁸ : StarRing R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : StarRing A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring B
      inst✝² : Algebra R B
      inst✝¹ : StarRing B
      inst✝ : StarModule R B
      S : Subalgebra R A
      a : A
      ha : Membership.mem (Algebra.adjoin R (Max.max ↑S ↑(Star.star S))) a
      ⊢ Membership.mem (Algebra.adjoin R (Star.star (Max.max ↑(Star.star S) ↑S))) a
    -/
    simpa using ha
    /-
      🎉 no goals
    -/


theorem starClosure_toSubalgebra (S : Subalgebra R A) : S.starClosure.toSubalgebra = S ⊔ star S :=
  rfl


theorem starClosure_le {S₁ : Subalgebra R A} {S₂ : StarSubalgebra R A} (h : S₁ ≤ S₂.toSubalgebra) :
    S₁.starClosure ≤ S₂ :=
  StarSubalgebra.toSubalgebra_le_iff.1 <|
    sup_le h fun x hx =>
      (star_star x ▸ star_mem (show star x ∈ S₂ from h <| (S₁.mem_star_iff _).1 hx) : x ∈ S₂)


theorem starClosure_le_iff {S₁ : Subalgebra R A} {S₂ : StarSubalgebra R A} :
    S₁.starClosure ≤ S₂ ↔ S₁ ≤ S₂.toSubalgebra :=
  ⟨fun h => le_sup_left.trans h, starClosure_le⟩


/-- The minimal star subalgebra that contains `s`. -/
@[simps!]
def adjoin (s : Set A) : StarSubalgebra R A :=
  { Algebra.adjoin R (s ∪ star s) with
    star_mem' := fun hx => by
      rwa [Subalgebra.mem_carrier, ← Subalgebra.mem_star_iff, Subalgebra.star_adjoin_comm,
        Set.union_star, star_star, Set.union_comm] }


theorem adjoin_eq_starClosure_adjoin (s : Set A) : adjoin R s = (Algebra.adjoin R s).starClosure :=
  toSubalgebra_injective <|
    show Algebra.adjoin R (s ∪ star s) = Algebra.adjoin R s ⊔ star (Algebra.adjoin R s) from
      (Subalgebra.star_adjoin_comm R s).symm ▸ Algebra.adjoin_union s (star s)


theorem adjoin_toSubalgebra (s : Set A) :
    (adjoin R s).toSubalgebra = Algebra.adjoin R (s ∪ star s) :=
  rfl


@[aesop safe 20 apply (rule_sets := [SetLike])]
theorem subset_adjoin (s : Set A) : s ⊆ adjoin R s :=
  Set.subset_union_left.trans Algebra.subset_adjoin


theorem star_subset_adjoin (s : Set A) : star s ⊆ adjoin R s :=
  Set.subset_union_right.trans Algebra.subset_adjoin


theorem self_mem_adjoin_singleton (x : A) : x ∈ adjoin R ({x} : Set A) :=
  Algebra.subset_adjoin <| Set.mem_union_left _ (Set.mem_singleton x)


theorem star_self_mem_adjoin_singleton (x : A) : star x ∈ adjoin R ({x} : Set A) :=
  star_mem <| self_mem_adjoin_singleton R x


protected theorem gc : GaloisConnection (adjoin R : Set A → StarSubalgebra R A) (↑) := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    ⊢ GaloisConnection (StarAlgebra.adjoin R) SetLike.coe
  -/
  intro s S
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    S : StarSubalgebra R A
    ⊢ Iff (LE.le (StarAlgebra.adjoin R s) S) (LE.le s ↑S)
  -/
  rw [← toSubalgebra_le_iff, adjoin_toSubalgebra, Algebra.adjoin_le_iff, coe_toSubalgebra]
  exact
    ⟨fun h => Set.subset_union_left.trans h, fun h =>
      Set.union_subset h fun x hx => star_star x ▸ star_mem (show star x ∈ S from h hx)⟩


/-- Galois insertion between `adjoin` and `coe`. -/
protected def gi : GaloisInsertion (adjoin R : Set A → StarSubalgebra R A) (↑) where
  choice s hs := (adjoin R s).copy s <| le_antisymm (StarAlgebra.gc.le_u_l s) hs
  gc := StarAlgebra.gc
  le_l_u S := (StarAlgebra.gc (S : Set A) (adjoin R S)).1 <| le_rfl
  choice_eq _ _ := StarSubalgebra.copy_eq _ _ _


theorem adjoin_le {S : StarSubalgebra R A} {s : Set A} (hs : s ⊆ S) : adjoin R s ≤ S :=
  StarAlgebra.gc.l_le hs


theorem adjoin_le_iff {S : StarSubalgebra R A} {s : Set A} : adjoin R s ≤ S ↔ s ⊆ S :=
  StarAlgebra.gc _ _


lemma adjoin_eq (S : StarSubalgebra R A) : adjoin R (S : Set A) = S :=
  le_antisymm (adjoin_le le_rfl) (subset_adjoin R (S : Set A))


open Submodule in
lemma adjoin_eq_span (s : Set A) :
    Subalgebra.toSubmodule (adjoin R s).toSubalgebra = span R (Submonoid.closure (s ∪ star s)) := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (Subalgebra.toSubmodule (StarAlgebra.adjoin R s).toSubalgebra) (Submodule …
  -/
  rw [adjoin_toSubalgebra, Algebra.adjoin_eq_span]
  /-
    🎉 no goals
  -/


theorem _root_.Subalgebra.starClosure_eq_adjoin (S : Subalgebra R A) :
    S.starClosure = adjoin R (S : Set A) :=
  le_antisymm (Subalgebra.starClosure_le_iff.2 <| subset_adjoin R (S : Set A))
    (adjoin_le (le_sup_left : S ≤ S ⊔ star S))


/-- If some predicate holds for all `x ∈ (s : Set A)` and this predicate is closed under the
`algebraMap`, addition, multiplication and star operations, then it holds for `a ∈ adjoin R s`. -/
@[elab_as_elim]
theorem adjoin_induction {s : Set A} {p : (x : A) → x ∈ adjoin R s → Prop}
    (mem : ∀ (x) (h : x ∈ s), p x (subset_adjoin R s h))
    (algebraMap : ∀ r, p (_root_.algebraMap R _ r) (_root_.algebraMap_mem _ r))
    (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    (star : ∀ x hx, p x hx → p (star x) (star_mem hx))
    {a : A} (ha : a ∈ adjoin R s) : p a ha := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    p : (x : A) → Membership.mem (StarAlgebra.adjoin R s) x → Prop
    mem : ∀ (x : A) (h : Membership.mem s x), p x ⋯
    algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
    add : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
    mul : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
    star : ∀ (x : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x), p x hx → p  …
    a : A
    ha : Membership.mem (StarAlgebra.adjoin R s) a
    ⊢ p a ha
  -/
  refine Algebra.adjoin_induction (fun x hx ↦ ?_) algebraMap add mul ha
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    p : (x : A) → Membership.mem (StarAlgebra.adjoin R s) x → Prop
    mem : ∀ (x : A) (h : Membership.mem s x), p x ⋯
    algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
    add : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
    mul : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
    star : ∀ (x : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x), p x hx → p  …
    a : A
    ha : Membership.mem (StarAlgebra.adjoin R s) a
    x : A
    hx : Membership.mem (Union.union s (Star.star s)) x
    ⊢ p x ⋯
  -/
  simp only [Set.mem_union, Set.mem_star] at hx
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    p : (x : A) → Membership.mem (StarAlgebra.adjoin R s) x → Prop
    mem : ∀ (x : A) (h : Membership.mem s x), p x ⋯
    algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
    add : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
    mul : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
    star : ∀ (x : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x), p x hx → p  …
    a : A
    ha : Membership.mem (StarAlgebra.adjoin R s) a
    x : A
    hx✝ : Membership.mem (Union.union s (Star.star s)) x
    hx : Or (Membership.mem s x) (Membership.mem s (Star.star x))
    ⊢ p x ⋯
  -/
  obtain (hx | hx) := hx
    /-
      case inl
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommSemiring R
      inst✝⁴ : StarRing R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : StarRing A
      inst✝ : StarModule R A
      s : Set A
      p : (x : A) → Membership.mem (StarAlgebra.adjoin R s) x → Prop
      mem : ∀ (x : A) (h : Membership.mem s x), p x ⋯
      algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
      add : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
      mul : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
      star : ∀ (x : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x), p x hx → p  …
      a : A
      ha : Membership.mem (StarAlgebra.adjoin R s) a
      x : A
      hx✝ : Membership.mem (Union.union s (Star.star s)) x
      hx : Membership.mem s x
      ⊢ p x ⋯
    -/
  · exact mem x hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_2
      A : Type u_3
      inst✝⁵ : CommSemiring R
      inst✝⁴ : StarRing R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : StarRing A
      inst✝ : StarModule R A
      s : Set A
      p : (x : A) → Membership.mem (StarAlgebra.adjoin R s) x → Prop
      mem : ∀ (x : A) (h : Membership.mem s x), p x ⋯
      algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
      add : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
      mul : ∀ (x y : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x) (hy : Membe …
      star : ∀ (x : A) (hx : Membership.mem (StarAlgebra.adjoin R s) x), p x hx → p  …
      a : A
      ha : Membership.mem (StarAlgebra.adjoin R s) a
      x : A
      hx✝ : Membership.mem (Union.union s (Star.star s)) x
      hx : Membership.mem s (Star.star x)
      ⊢ p x ⋯
    -/
  · simpa using star _ (Algebra.subset_adjoin (by simpa using Or.inl hx)) (mem _ hx)
    /-
      🎉 no goals
    -/


@[elab_as_elim]
theorem adjoin_induction₂ {s : Set A} {p : (x y : A) → x ∈ adjoin R s → y ∈ adjoin R s → Prop}
    (mem_mem : ∀ (x) (y) (hx : x ∈ s) (hy : y ∈ s), p x y (subset_adjoin R s hx)
      (subset_adjoin R s hy))
    (algebraMap_both : ∀ r₁ r₂, p (algebraMap R A r₁) (algebraMap R A r₂)
      (_root_.algebraMap_mem _ r₁) (_root_.algebraMap_mem _ r₂))
    (algebraMap_left : ∀ (r) (x) (hx : x ∈ s), p (algebraMap R A r) x (_root_.algebraMap_mem _ r)
      (subset_adjoin R s hx))
    (algebraMap_right : ∀ (r) (x) (hx : x ∈ s), p x (algebraMap R A r) (subset_adjoin R s hx)
      (_root_.algebraMap_mem _ r))
    (add_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x + y) z (add_mem hx hy) hz)
    (add_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y + z) hx (add_mem hy hz))
    (mul_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x * y) z (mul_mem hx hy) hz)
    (mul_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y * z) hx (mul_mem hy hz))
    (star_left : ∀ x y hx hy, p x y hx hy → p (star x) y (star_mem hx) hy)
    (star_right : ∀ x y hx hy, p x y hx hy → p x (star y) hx (star_mem hy))
    {a b : A} (ha : a ∈ adjoin R s) (hb : b ∈ adjoin R s) :
    p a b ha hb := by
  induction hb using adjoin_induction with
  | mem z hz => induction ha using adjoin_induction with
    | mem _ h => exact mem_mem _ _ h hz
    | algebraMap _ => exact algebraMap_left _ _ hz
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add_left _ _ _ _ _ _ h₁ h₂
    | star _ _ h => exact star_left _ _ _ _ h
  | algebraMap r =>
    induction ha using adjoin_induction with
    | mem _ h => exact algebraMap_right _ _ h
    | algebraMap _ => exact algebraMap_both _ _
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add_left _ _ _ _ _ _ h₁ h₂
    | star _ _ h => exact star_left _ _ _ _ h
  | mul _ _ _ _ h₁ h₂ => exact mul_right _ _ _ _ _ _ h₁ h₂
  | add _ _ _ _ h₁ h₂ => exact add_right _ _ _ _ _ _ h₁ h₂
  | star _ _ h => exact star_right _ _ _ _ h


/-- The difference with `StarSubalgebra.adjoin_induction` is that this acts on the subtype. -/
@[elab_as_elim]
theorem adjoin_induction_subtype {s : Set A} {p : adjoin R s → Prop} (a : adjoin R s)
    (mem : ∀ (x) (h : x ∈ s), p ⟨x, subset_adjoin R s h⟩) (algebraMap : ∀ r, p (algebraMap R _ r))
    (add : ∀ x y, p x → p y → p (x + y)) (mul : ∀ x y, p x → p y → p (x * y))
    (star : ∀ x, p x → p (star x)) : p a :=
  Subtype.recOn a fun b hb => by
    induction hb using adjoin_induction with
    | mem _ h => exact mem _ h
    | algebraMap _ => exact algebraMap _
    | mul _ _ _ _ h₁ h₂ => exact mul _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add _ _ h₁ h₂
    | star _ _ h => exact star _ h


lemma adjoin_le_centralizer_centralizer (s : Set A) :
    adjoin R s ≤ centralizer R (centralizer R s) := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    ⊢ LE.le (StarAlgebra.adjoin R s) (StarSubalgebra.centralizer R ↑(StarSubalgebr …
  -/
  rw [← toSubalgebra_le_iff, centralizer_toSubalgebra, adjoin_toSubalgebra]
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    ⊢ LE.le (Algebra.adjoin R (Union.union s (Star.star s))) (Subalgebra.centraliz …
  -/
  convert Algebra.adjoin_le_centralizer_centralizer R (s ∪ star s)
  /-
    case h.e'_4.h.e'_6
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (Union.union (↑(StarSubalgebra.centralizer R s)) (Star.star ↑(StarSubalge …
  -/
  rw [StarMemClass.star_coe_eq]
  /-
    case h.e'_4.h.e'_6
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    s : Set A
    ⊢ Eq (Union.union ↑(StarSubalgebra.centralizer R s) ↑(StarSubalgebra.centraliz …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If all elements of `s : Set A` commute pairwise and also commute pairwise with elements of
`star s`, then `StarSubalgebra.adjoin R s` is commutative. See note [reducible non-instances]. -/
abbrev adjoinCommSemiringOfComm {s : Set A}
    (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a)
    (hcomm_star : ∀ a ∈ s, ∀ b ∈ s, a * star b = star b * a) :
    CommSemiring (adjoin R s) :=
  { (adjoin R s).toSemiring with
    mul_comm := fun ⟨_, h₁⟩ ⟨_, h₂⟩ ↦ by
      have hcomm : ∀ a ∈ s ∪ star s, ∀ b ∈ s ∪ star s, a * b = b * a := fun a ha b hb ↦
        Set.union_star_self_comm (fun _ ha _ hb ↦ hcomm _ hb _ ha)
          (fun _ ha _ hb ↦ hcomm_star _ hb _ ha) b hb a ha
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring R
        inst✝⁸ : StarRing R
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : StarRing A
        inst✝⁴ : StarModule R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : StarRing B
        inst✝ : StarModule R B
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁ : Membership.mem (StarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂ : Membership.mem (StarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁⟩ ⟨val✝, h₂⟩) (HMul.hMul ⟨val✝, h₂⟩ ⟨val✝¹, h₁⟩)
      -/
      have := adjoin_le_centralizer_centralizer R s
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring R
        inst✝⁸ : StarRing R
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : StarRing A
        inst✝⁴ : StarModule R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : StarRing B
        inst✝ : StarModule R B
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁ : Membership.mem (StarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂ : Membership.mem (StarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (StarAlgebra.adjoin R s) (StarSubalgebra.centralizer R ↑(StarSuba …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁⟩ ⟨val✝, h₂⟩) (HMul.hMul ⟨val✝, h₂⟩ ⟨val✝¹, h₁⟩)
      -/
      apply this at h₁
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring R
        inst✝⁸ : StarRing R
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : StarRing A
        inst✝⁴ : StarModule R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : StarRing B
        inst✝ : StarModule R B
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁✝ : Membership.mem (StarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂ : Membership.mem (StarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (StarAlgebra.adjoin R s) (StarSubalgebra.centralizer R ↑(StarSuba …
        h₁ : Membership.mem (StarSubalgebra.centralizer R ↑(StarSubalgebra.centralizer …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁✝⟩ ⟨val✝, h₂⟩) (HMul.hMul ⟨val✝, h₂⟩ ⟨val✝¹, h₁✝⟩)
      -/
      apply this at h₂
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring R
        inst✝⁸ : StarRing R
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : StarRing A
        inst✝⁴ : StarModule R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : StarRing B
        inst✝ : StarModule R B
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁✝ : Membership.mem (StarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂✝ : Membership.mem (StarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (StarAlgebra.adjoin R s) (StarSubalgebra.centralizer R ↑(StarSuba …
        h₁ : Membership.mem (StarSubalgebra.centralizer R ↑(StarSubalgebra.centralizer …
        h₂ : Membership.mem (StarSubalgebra.centralizer R ↑(StarSubalgebra.centralizer …
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁✝⟩ ⟨val✝, h₂✝⟩) (HMul.hMul ⟨val✝, h₂✝⟩ ⟨val✝¹, h₁✝⟩)
      -/
      rw [← SetLike.mem_coe, coe_centralizer_centralizer] at h₁ h₂
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring R
        inst✝⁸ : StarRing R
        inst✝⁷ : Semiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : StarRing A
        inst✝⁴ : StarModule R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : StarRing B
        inst✝ : StarModule R B
        s : Set A
        hcomm✝ : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → Eq (H …
        hcomm_star : ∀ (a : A), Membership.mem s a → ∀ (b : A), Membership.mem s b → E …
        x✝¹ x✝ : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
        val✝¹ : A
        h₁✝ : Membership.mem (StarAlgebra.adjoin R s) val✝¹
        val✝ : A
        h₂✝ : Membership.mem (StarAlgebra.adjoin R s) val✝
        hcomm : ∀ (a : A), Membership.mem (Union.union s (Star.star s)) a → ∀ (b : A), …
        this : LE.le (StarAlgebra.adjoin R s) (StarSubalgebra.centralizer R ↑(StarSuba …
        h₁ : Membership.mem (Union.union s (Star.star s)).centralizer.centralizer val✝¹
        h₂ : Membership.mem (Union.union s (Star.star s)).centralizer.centralizer val✝
        ⊢ Eq (HMul.hMul ⟨val✝¹, h₁✝⟩ ⟨val✝, h₂✝⟩) (HMul.hMul ⟨val✝, h₂✝⟩ ⟨val✝¹, h₁✝⟩)
      -/
      exact Subtype.ext <| Set.centralizer_centralizer_comm_of_comm hcomm _ h₁ _ h₂ }
      /-
        🎉 no goals
      -/


/-- If all elements of `s : Set A` commute pairwise and also commute pairwise with elements of
`star s`, then `StarSubalgebra.adjoin R s` is commutative. See note [reducible non-instances]. -/
abbrev adjoinCommRingOfComm (R : Type u) {A : Type v} [CommRing R] [StarRing R] [Ring A]
    [Algebra R A] [StarRing A] [StarModule R A] {s : Set A}
    (hcomm : ∀ a : A, a ∈ s → ∀ b : A, b ∈ s → a * b = b * a)
    (hcomm_star : ∀ a : A, a ∈ s → ∀ b : A, b ∈ s → a * star b = star b * a) :
    CommRing (adjoin R s) :=
  { StarAlgebra.adjoinCommSemiringOfComm R hcomm hcomm_star,
    (adjoin R s).toSubalgebra.toRing with }


/-- The star subalgebra `StarSubalgebra.adjoin R {x}` generated by a single `x : A` is commutative
if `x` is normal. -/
instance adjoinCommSemiringOfIsStarNormal (x : A) [IsStarNormal x] :
    CommSemiring (adjoin R ({x} : Set A)) :=
  adjoinCommSemiringOfComm R
    (fun a ha b hb => by
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : StarRing R
        inst✝⁸ : Semiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : StarRing A
        inst✝⁵ : StarModule R A
        inst✝⁴ : Semiring B
        inst✝³ : Algebra R B
        inst✝² : StarRing B
        inst✝¹ : StarModule R B
        x : A
        inst✝ : IsStarNormal x
        a : A
        ha : Membership.mem (Singleton.singleton x) a
        b : A
        hb : Membership.mem (Singleton.singleton x) b
        ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      rw [Set.mem_singleton_iff] at ha hb
      /-
        F : Type u_1
        R : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : StarRing R
        inst✝⁸ : Semiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : StarRing A
        inst✝⁵ : StarModule R A
        inst✝⁴ : Semiring B
        inst✝³ : Algebra R B
        inst✝² : StarRing B
        inst✝¹ : StarModule R B
        x : A
        inst✝ : IsStarNormal x
        a : A
        ha : Eq a x
        b : A
        hb : Eq b x
        ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      rw [ha, hb])
      /-
        🎉 no goals
      -/
    fun a ha b hb => by
    /-
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : StarRing R
      inst✝⁸ : Semiring A
      inst✝⁷ : Algebra R A
      inst✝⁶ : StarRing A
      inst✝⁵ : StarModule R A
      inst✝⁴ : Semiring B
      inst✝³ : Algebra R B
      inst✝² : StarRing B
      inst✝¹ : StarModule R B
      x : A
      inst✝ : IsStarNormal x
      a : A
      ha : Membership.mem (Singleton.singleton x) a
      b : A
      hb : Membership.mem (Singleton.singleton x) b
      ⊢ Eq (HMul.hMul a (Star.star b)) (HMul.hMul (Star.star b) a)
    -/
    rw [Set.mem_singleton_iff] at ha hb
    /-
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : StarRing R
      inst✝⁸ : Semiring A
      inst✝⁷ : Algebra R A
      inst✝⁶ : StarRing A
      inst✝⁵ : StarModule R A
      inst✝⁴ : Semiring B
      inst✝³ : Algebra R B
      inst✝² : StarRing B
      inst✝¹ : StarModule R B
      x : A
      inst✝ : IsStarNormal x
      a : A
      ha : Eq a x
      b : A
      hb : Eq b x
      ⊢ Eq (HMul.hMul a (Star.star b)) (HMul.hMul (Star.star b) a)
    -/
    simpa only [ha, hb] using (star_comm_self' x).symm
    /-
      🎉 no goals
    -/


/-- The star subalgebra `StarSubalgebra.adjoin R {x}` generated by a single `x : A` is commutative
if `x` is normal. -/
instance adjoinCommRingOfIsStarNormal (R : Type u) {A : Type v} [CommRing R] [StarRing R] [Ring A]
    [Algebra R A] [StarRing A] [StarModule R A] (x : A) [IsStarNormal x] :
    CommRing (adjoin R ({x} : Set A)) :=
  { (adjoin R ({x} : Set A)).toSubalgebra.toRing with mul_comm := mul_comm }


instance completeLattice : CompleteLattice (StarSubalgebra R A) where
  __ := GaloisInsertion.liftCompleteLattice StarAlgebra.gi
  bot := { toSubalgebra := ⊥, star_mem' := fun ⟨r, hr⟩ => ⟨star r, hr ▸ algebraMap_star_comm _⟩ }
  bot_le S := (bot_le : ⊥ ≤ S.toSubalgebra)


instance inhabited : Inhabited (StarSubalgebra R A) :=
  ⟨⊤⟩


@[simp]
theorem coe_top : (↑(⊤ : StarSubalgebra R A) : Set A) = Set.univ :=
  rfl


@[simp]
theorem mem_top {x : A} : x ∈ (⊤ : StarSubalgebra R A) :=
  Set.mem_univ x


@[simp]
                                                                           /-
                                                                             R : Type u_2
                                                                             A : Type u_3
                                                                             inst✝⁵ : CommSemiring R
                                                                             inst✝⁴ : StarRing R
                                                                             inst✝³ : Semiring A
                                                                             inst✝² : Algebra R A
                                                                             inst✝¹ : StarRing A
                                                                             inst✝ : StarModule R A
                                                                             ⊢ Eq Top.top.toSubalgebra Top.top
                                                                           -/
theorem top_toSubalgebra : (⊤ : StarSubalgebra R A).toSubalgebra = ⊤ := by ext; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
-- Porting note: Lean can no longer prove this by `rfl`, it times out


@[simp]
theorem toSubalgebra_eq_top {S : StarSubalgebra R A} : S.toSubalgebra = ⊤ ↔ S = ⊤ :=
  StarSubalgebra.toSubalgebra_injective.eq_iff' top_toSubalgebra


theorem mem_sup_left {S T : StarSubalgebra R A} : ∀ {x : A}, x ∈ S → x ∈ S ⊔ T :=
  have : S ≤ S ⊔ T := le_sup_left; (this ·) -- Porting note: need `have` instead of `show`


theorem mem_sup_right {S T : StarSubalgebra R A} : ∀ {x : A}, x ∈ T → x ∈ S ⊔ T :=
  have : T ≤ S ⊔ T := le_sup_right; (this ·) -- Porting note: need `have` instead of `show`


theorem mul_mem_sup {S T : StarSubalgebra R A} {x y : A} (hx : x ∈ S) (hy : y ∈ T) :
    x * y ∈ S ⊔ T :=
  mul_mem (mem_sup_left hx) (mem_sup_right hy)


theorem map_sup (f : A →⋆ₐ[R] B) (S T : StarSubalgebra R A) : map f (S ⊔ T) = map f S ⊔ map f T :=
  (StarSubalgebra.gc_map_comap f).l_sup


theorem map_inf (f : A →⋆ₐ[R] B) (hf : Function.Injective f) (S T : StarSubalgebra R A) :
    map f (S ⊓ T) = map f S ⊓ map f T := SetLike.coe_injective (Set.image_inter hf)


@[simp, norm_cast]
theorem coe_inf (S T : StarSubalgebra R A) : (↑(S ⊓ T) : Set A) = (S : Set A) ∩ T :=
  rfl


@[simp]
theorem mem_inf {S T : StarSubalgebra R A} {x : A} : x ∈ S ⊓ T ↔ x ∈ S ∧ x ∈ T :=
  Iff.rfl


@[simp]
theorem inf_toSubalgebra (S T : StarSubalgebra R A) :
    (S ⊓ T).toSubalgebra = S.toSubalgebra ⊓ T.toSubalgebra := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    S T : StarSubalgebra R A
    ⊢ Eq (Min.min S T).toSubalgebra (Min.min S.toSubalgebra T.toSubalgebra)
  -/
  ext; simp
       /-
         🎉 no goals
       -/
-- Porting note: Lean can no longer prove this by `rfl`, it times out


@[simp, norm_cast]
theorem coe_sInf (S : Set (StarSubalgebra R A)) : (↑(sInf S) : Set A) = ⋂ s ∈ S, ↑s :=
  sInf_image


theorem mem_sInf {S : Set (StarSubalgebra R A)} {x : A} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    S : Set (StarSubalgebra R A)
    x : A
    ⊢ Iff (Membership.mem (InfSet.sInf S) x) (∀ (p : StarSubalgebra R A), Membersh …
  -/
  simp only [← SetLike.mem_coe, coe_sInf, Set.mem_iInter₂]
  /-
    🎉 no goals
  -/


@[simp]
theorem sInf_toSubalgebra (S : Set (StarSubalgebra R A)) :
    (sInf S).toSubalgebra = sInf (StarSubalgebra.toSubalgebra '' S) :=
                              /-
                                R : Type u_2
                                A : Type u_3
                                inst✝⁵ : CommSemiring R
                                inst✝⁴ : StarRing R
                                inst✝³ : Semiring A
                                inst✝² : Algebra R A
                                inst✝¹ : StarRing A
                                inst✝ : StarModule R A
                                S : Set (StarSubalgebra R A)
                                ⊢ Eq ↑(InfSet.sInf S).toSubalgebra ↑(InfSet.sInf (Set.image StarSubalgebra.toS …
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} {S : ι → StarSubalgebra R A} : (↑(⨅ i, S i) : Set A) = ⋂ i, S i := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    ι : Sort u_5
    S : ι → StarSubalgebra R A
    ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
  -/
  simp [iInf]
  /-
    🎉 no goals
  -/


theorem mem_iInf {ι : Sort*} {S : ι → StarSubalgebra R A} {x : A} :
                                        /-
                                          R : Type u_2
                                          A : Type u_3
                                          inst✝⁵ : CommSemiring R
                                          inst✝⁴ : StarRing R
                                          inst✝³ : Semiring A
                                          inst✝² : Algebra R A
                                          inst✝¹ : StarRing A
                                          inst✝ : StarModule R A
                                          ι : Sort u_5
                                          S : ι → StarSubalgebra R A
                                          x : A
                                          ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
                                        -/
    (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by simp only [iInf, mem_sInf, Set.forall_mem_range]
                                        /-
                                          🎉 no goals
                                        -/


theorem map_iInf {ι : Sort*} [Nonempty ι] (f : A →⋆ₐ[R] B) (hf : Function.Injective f)
    (s : ι → StarSubalgebra R A) : map f (iInf s) = ⨅ (i : ι), map f (s i) := by
  /-
    R : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : StarRing R
    inst✝⁸ : Semiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : StarRing A
    inst✝⁵ : StarModule R A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R B
    inst✝² : StarRing B
    inst✝¹ : StarModule R B
    ι : Sort u_5
    inst✝ : Nonempty ι
    f : StarAlgHom R A B
    hf : Function.Injective ⇑f
    s : ι → StarSubalgebra R A
    ⊢ Eq (StarSubalgebra.map f (iInf s)) (iInf fun i => StarSubalgebra.map f (s i))
  -/
  apply SetLike.coe_injective
  /-
    case a
    R : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : StarRing R
    inst✝⁸ : Semiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : StarRing A
    inst✝⁵ : StarModule R A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R B
    inst✝² : StarRing B
    inst✝¹ : StarModule R B
    ι : Sort u_5
    inst✝ : Nonempty ι
    f : StarAlgHom R A B
    hf : Function.Injective ⇑f
    s : ι → StarSubalgebra R A
    ⊢ Eq ↑(StarSubalgebra.map f (iInf s)) ↑(iInf fun i => StarSubalgebra.map f (s  …
  -/
  simpa using (Set.injOn_of_injective hf).image_iInter_eq (s := SetLike.coe ∘ s)
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_toSubalgebra {ι : Sort*} (S : ι → StarSubalgebra R A) :
    (⨅ i, S i).toSubalgebra = ⨅ i, (S i).toSubalgebra :=
                              /-
                                R : Type u_2
                                A : Type u_3
                                inst✝⁵ : CommSemiring R
                                inst✝⁴ : StarRing R
                                inst✝³ : Semiring A
                                inst✝² : Algebra R A
                                inst✝¹ : StarRing A
                                inst✝ : StarModule R A
                                ι : Sort u_5
                                S : ι → StarSubalgebra R A
                                ⊢ Eq ↑(iInf fun i => S i).toSubalgebra ↑(iInf fun i => (S i).toSubalgebra)
                              -/
  SetLike.coe_injective <| by simp
                              /-
                                🎉 no goals
                              -/


theorem bot_toSubalgebra : (⊥ : StarSubalgebra R A).toSubalgebra = ⊥ := rfl


theorem mem_bot {x : A} : x ∈ (⊥ : StarSubalgebra R A) ↔ x ∈ Set.range (algebraMap R A) := Iff.rfl


@[simp]
theorem coe_bot : ((⊥ : StarSubalgebra R A) : Set A) = Set.range (algebraMap R A) := rfl


theorem eq_top_iff {S : StarSubalgebra R A} : S = ⊤ ↔ ∀ x : A, x ∈ S :=
                 /-
                   R : Type u_2
                   A : Type u_3
                   inst✝⁵ : CommSemiring R
                   inst✝⁴ : StarRing R
                   inst✝³ : Semiring A
                   inst✝² : Algebra R A
                   inst✝¹ : StarRing A
                   inst✝ : StarModule R A
                   S : StarSubalgebra R A
                   h : Eq S Top.top
                   x : A
                   ⊢ Membership.mem S x
                 -/
  ⟨fun h x => by rw [h]; exact mem_top,
                         /-
                           🎉 no goals
                         -/
              /-
                R : Type u_2
                A : Type u_3
                inst✝⁵ : CommSemiring R
                inst✝⁴ : StarRing R
                inst✝³ : Semiring A
                inst✝² : Algebra R A
                inst✝¹ : StarRing A
                inst✝ : StarModule R A
                S : StarSubalgebra R A
                h : ∀ (x : A), Membership.mem S x
                ⊢ Eq S Top.top
              -/
  fun h => by ext x; exact ⟨fun _ => mem_top, fun _ => h x⟩⟩
                     /-
                       🎉 no goals
                     -/


theorem ext_adjoin {s : Set A} [FunLike F (adjoin R s) B]
    [AlgHomClass F R (adjoin R s) B] [StarHomClass F (adjoin R s) B] {f g : F}
    (h : ∀ x : adjoin R s, (x : A) ∈ s → f x = g x) : f = g := by
  refine DFunLike.ext f g fun a =>
    adjoin_induction_subtype (p := fun y => f y = g y) a (fun x hx => ?_) (fun r => ?_)
    (fun x y hx hy => ?_) (fun x y hx hy => ?_) fun x hx => ?_
    /-
      case refine_1
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : StarRing A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : StarRing B
      inst✝³ : StarModule R A
      s : Set A
      inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x …
      inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.adjoin  …
      inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R  …
      f g : F
      h : ∀ (x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x), Member …
      a : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
      x : A
      hx : Membership.mem s x
      ⊢ (fun y => Eq (f y) (g y)) ⟨x, ⋯⟩
    -/
  · exact h ⟨x, subset_adjoin R s hx⟩ hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : StarRing A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : StarRing B
      inst✝³ : StarModule R A
      s : Set A
      inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x …
      inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.adjoin  …
      inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R  …
      f g : F
      h : ∀ (x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x), Member …
      a : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
      r : R
      ⊢ (fun y => Eq (f y) (g y)) ((algebraMap R (Subtype fun x => Membership.mem (S …
    -/
  · simp only [AlgHomClass.commutes]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : StarRing A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : StarRing B
      inst✝³ : StarModule R A
      s : Set A
      inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x …
      inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.adjoin  …
      inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R  …
      f g : F
      h : ∀ (x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x), Member …
      a x y : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
      hx : (fun y => Eq (f y) (g y)) x
      hy : (fun y => Eq (f y) (g y)) y
      ⊢ (fun y => Eq (f y) (g y)) (HAdd.hAdd x y)
    -/
  · simp only [map_add, map_add, hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : StarRing A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : StarRing B
      inst✝³ : StarModule R A
      s : Set A
      inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x …
      inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.adjoin  …
      inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R  …
      f g : F
      h : ∀ (x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x), Member …
      a x y : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
      hx : (fun y => Eq (f y) (g y)) x
      hy : (fun y => Eq (f y) (g y)) y
      ⊢ (fun y => Eq (f y) (g y)) (HMul.hMul x y)
    -/
  · simp only [map_mul, map_mul, hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case refine_5
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : StarRing R
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : StarRing A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : StarRing B
      inst✝³ : StarModule R A
      s : Set A
      inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x …
      inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.adjoin  …
      inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.adjoin R  …
      f g : F
      h : ∀ (x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x), Member …
      a x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R s) x
      hx : (fun y => Eq (f y) (g y)) x
      ⊢ (fun y => Eq (f y) (g y)) (Star.star x)
    -/
  · simp only [map_star, hx]
    /-
      🎉 no goals
    -/


theorem ext_adjoin_singleton {a : A} [FunLike F (adjoin R ({a} : Set A)) B]
    [AlgHomClass F R (adjoin R ({a} : Set A)) B] [StarHomClass F (adjoin R ({a} : Set A)) B]
    {f g : F} (h : f ⟨a, self_mem_adjoin_singleton R a⟩ = g ⟨a, self_mem_adjoin_singleton R a⟩) :
    f = g :=
  ext_adjoin fun x hx =>
    (show x = ⟨a, self_mem_adjoin_singleton R a⟩ from
          Subtype.ext <| Set.mem_singleton_iff.mp hx).symm ▸
      h


/-- The equalizer of two star `R`-algebra homomorphisms. -/
def equalizer : StarSubalgebra R A where
  toSubalgebra := AlgHom.equalizer (f : A →ₐ[R] B) g
                                       /-
                                         F : Type u_1
                                         R : Type u_2
                                         A : Type u_3
                                         B : Type u_4
                                         inst✝¹¹ : CommSemiring R
                                         inst✝¹⁰ : StarRing R
                                         inst✝⁹ : Semiring A
                                         inst✝⁸ : Algebra R A
                                         inst✝⁷ : StarRing A
                                         inst✝⁶ : Semiring B
                                         inst✝⁵ : Algebra R B
                                         inst✝⁴ : StarRing B
                                         inst✝³ : StarModule R A
                                         inst✝² : FunLike F A B
                                         inst✝¹ : AlgHomClass F R A B
                                         inst✝ : StarHomClass F A B
                                         f g : F
                                         a : A
                                         ha : Eq (f a) (g a)
                                         ⊢ Membership.mem (AlgHom.equalizer ↑f ↑g).carrier (Star.star a)
                                       -/
  star_mem' {a} (ha : f a = g a) := by simpa only [← map_star] using congrArg star ha
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem mem_equalizer (x : A) : x ∈ StarAlgHom.equalizer f g ↔ f x = g x :=
  Iff.rfl


theorem adjoin_le_equalizer {s : Set A} (h : s.EqOn f g) : adjoin R s ≤ StarAlgHom.equalizer f g :=
  adjoin_le h


theorem ext_of_adjoin_eq_top {s : Set A} (h : adjoin R s = ⊤) ⦃f g : F⦄ (hs : s.EqOn f g) : f = g :=
  DFunLike.ext f g fun _x => StarAlgHom.adjoin_le_equalizer f g hs <| h.symm ▸ trivial



theorem map_adjoin (f : A →⋆ₐ[R] B) (s : Set A) :
    map f (adjoin R s) = adjoin R (f '' s) :=
  GaloisConnection.l_comm_of_u_comm Set.image_preimage (gc_map_comap f) StarAlgebra.gc
    StarAlgebra.gc fun _ => rfl


/-- Range of a `StarAlgHom` as a star subalgebra. -/
protected def range
    (φ : A →⋆ₐ[R] B) : StarSubalgebra R B where
  toSubalgebra := φ.toAlgHom.range
                  /-
                    F : Type u_1
                    R : Type u_2
                    A : Type u_3
                    B : Type u_4
                    inst✝¹² : CommSemiring R
                    inst✝¹¹ : StarRing R
                    inst✝¹⁰ : Semiring A
                    inst✝⁹ : Algebra R A
                    inst✝⁸ : StarRing A
                    inst✝⁷ : Semiring B
                    inst✝⁶ : Algebra R B
                    inst✝⁵ : StarRing B
                    inst✝⁴ : StarModule R A
                    inst✝³ : FunLike F A B
                    inst✝² : AlgHomClass F R A B
                    inst✝¹ : StarHomClass F A B
                    f g : F
                    inst✝ : StarModule R B
                    φ : StarAlgHom R A B
                    ⊢ ∀ {a : B}, Membership.mem φ.range.carrier a → Membership.mem φ.range.carrier …
                  -/
  star_mem' := by rintro _ ⟨b, rfl⟩; exact ⟨star b, map_star φ b⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem range_eq_map_top (φ : A →⋆ₐ[R] B) : φ.range = (⊤ : StarSubalgebra R A).map φ :=
  StarSubalgebra.ext fun x =>
        /-
          R : Type u_2
          A : Type u_3
          B : Type u_4
          inst✝⁹ : CommSemiring R
          inst✝⁸ : StarRing R
          inst✝⁷ : Semiring A
          inst✝⁶ : Algebra R A
          inst✝⁵ : StarRing A
          inst✝⁴ : Semiring B
          inst✝³ : Algebra R B
          inst✝² : StarRing B
          inst✝¹ : StarModule R A
          inst✝ : StarModule R B
          φ : StarAlgHom R A B
          x : B
          ⊢ Membership.mem φ.range x → Membership.mem (StarSubalgebra.map φ Top.top) x
        -/
                        /-
                          🎉 no goals
                        -/
    ⟨by rintro ⟨a, ha⟩; exact ⟨a, by simp, ha⟩, by rintro ⟨a, -, ha⟩; exact ⟨a, ha⟩⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- Restriction of the codomain of a `StarAlgHom` to a star subalgebra containing the range. -/
protected def codRestrict (f : A →⋆ₐ[R] B) (S : StarSubalgebra R B) (hf : ∀ x, f x ∈ S) :
    A →⋆ₐ[R] S where
  toAlgHom := AlgHom.codRestrict f.toAlgHom S.toSubalgebra hf
  map_star' := fun x => Subtype.ext (map_star f x)


@[simp]
theorem coe_codRestrict (f : A →⋆ₐ[R] B) (S : StarSubalgebra R B) (hf : ∀ x, f x ∈ S) (x : A) :
    ↑(f.codRestrict S hf x) = f x :=
  rfl


@[simp]
theorem subtype_comp_codRestrict (f : A →⋆ₐ[R] B) (S : StarSubalgebra R B)
    (hf : ∀ x : A, f x ∈ S) : S.subtype.comp (f.codRestrict S hf) = f :=
  StarAlgHom.ext <| coe_codRestrict _ S hf


theorem injective_codRestrict (f : A →⋆ₐ[R] B) (S : StarSubalgebra R B) (hf : ∀ x : A, f x ∈ S) :
    Function.Injective (StarAlgHom.codRestrict f S hf) ↔ Function.Injective f :=
  ⟨fun H _x _y hxy => H <| Subtype.eq hxy, fun H _x _y hxy => H (congr_arg Subtype.val hxy : _)⟩


/-- Restriction of the codomain of a `StarAlgHom` to its range. -/
def rangeRestrict (f : A →⋆ₐ[R] B) : A →⋆ₐ[R] f.range :=
  StarAlgHom.codRestrict f _ fun x => ⟨x, rfl⟩


/-- The `StarAlgEquiv` onto the range corresponding to an injective `StarAlgHom`. -/
@[simps]
noncomputable def _root_.StarAlgEquiv.ofInjective (f : A →⋆ₐ[R] B)
    (hf : Function.Injective f) : A ≃⋆ₐ[R] f.range :=
  { AlgEquiv.ofInjective (f : A →ₐ[R] B) hf with
    toFun := f.rangeRestrict
    map_star' := fun a => Subtype.ext (map_star f a)
    map_smul' := fun r a => Subtype.ext (map_smul f r a) }

@[simps!]
def StarAlgHom.restrictScalars (f : A →⋆ₐ[S] B) : A →⋆ₐ[R] B where
  toAlgHom := f.toAlgHom.restrictScalars R
  map_star' := map_star f


theorem StarAlgHom.restrictScalars_injective :
    Function.Injective (StarAlgHom.restrictScalars R : (A →⋆ₐ[S] B) → A →⋆ₐ[R] B) :=
  fun f g h => StarAlgHom.ext fun x =>
    show f.restrictScalars R x = g.restrictScalars R x from DFunLike.congr_fun h x


@[simps]
def StarAlgEquiv.restrictScalars (f : A ≃⋆ₐ[S] B) : A ≃⋆ₐ[R] B :=
  { (f : A →⋆ₐ[S] B).restrictScalars R, f with
    toFun := f
    map_smul' := map_smul ((f : A →⋆ₐ[S] B).restrictScalars R) }


theorem StarAlgEquiv.restrictScalars_injective :
    Function.Injective (StarAlgEquiv.restrictScalars R : (A ≃⋆ₐ[S] B) → A ≃⋆ₐ[R] B) :=
  fun f g h => StarAlgEquiv.ext fun x =>
    show f.restrictScalars R x = g.restrictScalars R x from DFunLike.congr_fun h x


