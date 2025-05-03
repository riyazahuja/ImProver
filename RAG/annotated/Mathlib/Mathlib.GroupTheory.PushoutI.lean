/-- The relation we quotient by to form the pushout -/
def PushoutI.con [∀ i, Monoid (G i)] [Monoid H] (φ : ∀ i, H →* G i) :
    Con (Coprod (CoprodI G) H) :=
  conGen (fun x y : Coprod (CoprodI G) H =>
    ∃ i x', x = inl (of (φ i x')) ∧ y = inr x')


/-- The indexed pushout of monoids, which is the pushout in the category of monoids,
or the category of groups. -/
def PushoutI [∀ i, Monoid (G i)] [Monoid H] (φ : ∀ i, H →* G i) : Type _ :=
  (PushoutI.con φ).Quotient


protected instance mul : Mul (PushoutI φ) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝² : Monoid K
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    ⊢ Mul (Monoid.PushoutI φ)
  -/
  delta PushoutI; infer_instance
                  /-
                    🎉 no goals
                  -/


protected instance one : One (PushoutI φ) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝² : Monoid K
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    ⊢ One (Monoid.PushoutI φ)
  -/
  delta PushoutI; infer_instance
                  /-
                    🎉 no goals
                  -/


instance monoid : Monoid (PushoutI φ) :=
  { Con.monoid _ with
    toMul := PushoutI.mul
    toOne := PushoutI.one }


/-- The map from each indexing group into the pushout -/
def of (i : ι) : G i →* PushoutI φ :=
  (Con.mk' _).comp <| inl.comp CoprodI.of


variable (φ) in
/-- The map from the base monoid into the pushout -/
def base : H →* PushoutI φ :=
  (Con.mk' _).comp inr


theorem of_comp_eq_base (i : ι) : (of i).comp (φ i) = (base φ) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    ⊢ Eq ((Monoid.PushoutI.of i).comp (φ i)) (Monoid.PushoutI.base φ)
  -/
  ext x
  /-
    case h
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    x : H
    ⊢ Eq (((Monoid.PushoutI.of i).comp (φ i)) x) ((Monoid.PushoutI.base φ) x)
  -/
  apply (Con.eq _).2
  /-
    case h
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    x : H
    ⊢ (Monoid.PushoutI.con φ) ((Monoid.Coprod.inl.comp Monoid.CoprodI.of) ((φ i) x …
  -/
  refine ConGen.Rel.of _ _ ?_
  /-
    case h
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    x : H
    ⊢ Exists fun i_1 => Exists fun x' => And (Eq ((Monoid.Coprod.inl.comp Monoid.C …
  -/
  simp only [MonoidHom.comp_apply, Set.mem_iUnion, Set.mem_range]
  /-
    case h
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    x : H
    ⊢ Exists fun i_1 => Exists fun x' => And (Eq (Monoid.Coprod.inl (Monoid.Coprod …
  -/
  exact ⟨_, _, rfl, rfl⟩
  /-
    🎉 no goals
  -/


variable (φ) in
theorem of_apply_eq_base (i : ι) (x : H) : of i (φ i x) = base φ x := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    x : H
    ⊢ Eq ((Monoid.PushoutI.of i) ((φ i) x)) ((Monoid.PushoutI.base φ) x)
  -/
  rw [← MonoidHom.comp_apply, of_comp_eq_base]
  /-
    🎉 no goals
  -/


/-- Define a homomorphism out of the pushout of monoids be defining it on each object in the
diagram -/
def lift (f : ∀ i, G i →* K) (k : H →* K)
    (hf : ∀ i, (f i).comp (φ i) = k) :
    PushoutI φ →* K :=
  Con.lift _ (Coprod.lift (CoprodI.lift f) k) <| by
    /-
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝² : Monoid K
      inst✝¹ : (i : ι) → Monoid (G i)
      inst✝ : Monoid H
      φ : (i : ι) → MonoidHom H (G i)
      f : (i : ι) → MonoidHom (G i) K
      k : MonoidHom H K
      hf : ∀ (i : ι), Eq ((f i).comp (φ i)) k
      ⊢ LE.le (Monoid.PushoutI.con φ) (Con.ker (Monoid.Coprod.lift (Monoid.CoprodI.l …
    -/
    apply Con.conGen_le fun x y => ?_
    /-
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝² : Monoid K
      inst✝¹ : (i : ι) → Monoid (G i)
      inst✝ : Monoid H
      φ : (i : ι) → MonoidHom H (G i)
      f : (i : ι) → MonoidHom (G i) K
      k : MonoidHom H K
      hf : ∀ (i : ι), Eq ((f i).comp (φ i)) k
      x y : Monoid.Coprod (Monoid.CoprodI G) H
      ⊢ (Exists fun i => Exists fun x' => And (Eq x (Monoid.Coprod.inl (Monoid.Copro …
    -/
    rintro ⟨i, x', rfl, rfl⟩
    /-
      case intro.intro.intro
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝² : Monoid K
      inst✝¹ : (i : ι) → Monoid (G i)
      inst✝ : Monoid H
      φ : (i : ι) → MonoidHom H (G i)
      f : (i : ι) → MonoidHom (G i) K
      k : MonoidHom H K
      hf : ∀ (i : ι), Eq ((f i).comp (φ i)) k
      i : ι
      x' : H
      ⊢ (Con.ker (Monoid.Coprod.lift (Monoid.CoprodI.lift f) k)) (Monoid.Coprod.inl  …
    -/
    simp only [DFunLike.ext_iff, MonoidHom.coe_comp, comp_apply] at hf
    /-
      case intro.intro.intro
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝² : Monoid K
      inst✝¹ : (i : ι) → Monoid (G i)
      inst✝ : Monoid H
      φ : (i : ι) → MonoidHom H (G i)
      f : (i : ι) → MonoidHom (G i) K
      k : MonoidHom H K
      i : ι
      x' : H
      hf : ∀ (i : ι) (x : H), Eq ((f i) ((φ i) x)) (k x)
      ⊢ (Con.ker (Monoid.Coprod.lift (Monoid.CoprodI.lift f) k)) (Monoid.Coprod.inl  …
    -/
    simp [hf]
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_of (f : ∀ i, G i →* K) (k : H →* K)
    (hf : ∀ i, (f i).comp (φ i) = k)
    {i : ι} (g : G i) : (lift f k hf) (of i g : PushoutI φ) = f i g := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝² : Monoid K
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    f : (i : ι) → MonoidHom (G i) K
    k : MonoidHom H K
    hf : ∀ (i : ι), Eq ((f i).comp (φ i)) k
    i : ι
    g : G i
    ⊢ Eq ((Monoid.PushoutI.lift f k hf) ((Monoid.PushoutI.of i) g)) ((f i) g)
  -/
  delta PushoutI lift of
  simp only [MonoidHom.coe_comp, Con.coe_mk', comp_apply, Con.lift_coe,
    lift_apply_inl, CoprodI.lift_of]


@[simp]
theorem lift_base (f : ∀ i, G i →* K) (k : H →* K)
    (hf : ∀ i, (f i).comp (φ i) = k)
    (g : H) : (lift f k hf) (base φ g : PushoutI φ) = k g := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝² : Monoid K
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    f : (i : ι) → MonoidHom (G i) K
    k : MonoidHom H K
    hf : ∀ (i : ι), Eq ((f i).comp (φ i)) k
    g : H
    ⊢ Eq ((Monoid.PushoutI.lift f k hf) ((Monoid.PushoutI.base φ) g)) (k g)
  -/
  delta PushoutI lift base
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝² : Monoid K
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    f : (i : ι) → MonoidHom (G i) K
    k : MonoidHom H K
    hf : ∀ (i : ι), Eq ((f i).comp (φ i)) k
    g : H
    ⊢ Eq (((Monoid.PushoutI.con φ).lift (Monoid.Coprod.lift (Monoid.CoprodI.lift f …
  -/
  simp only [MonoidHom.coe_comp, Con.coe_mk', comp_apply, Con.lift_coe, lift_apply_inr]
  /-
    🎉 no goals
  -/

-- `ext` attribute should be lower priority then `hom_ext_nonempty`

@[ext 1199]
theorem hom_ext {f g : PushoutI φ →* K}
    (h : ∀ i, f.comp (of i : G i →* _) = g.comp (of i : G i →* _))
    (hbase : f.comp (base φ) = g.comp (base φ)) : f = g :=
  (MonoidHom.cancel_right Con.mk'_surjective).mp <|
    Coprod.hom_ext
      (CoprodI.ext_hom _ _ h)
      hbase


@[ext high]
theorem hom_ext_nonempty [hn : Nonempty ι]
    {f g : PushoutI φ →* K}
    (h : ∀ i, f.comp (of i : G i →* _) = g.comp (of i : G i →* _)) : f = g :=
  hom_ext h <| by
    cases hn with
    | intro i =>
      ext
      rw [← of_comp_eq_base i, ← MonoidHom.comp_assoc, h, MonoidHom.comp_assoc]


/-- The equivalence that is part of the universal property of the pushout. A hom out of
the pushout is just a morphism out of all groups in the pushout that satisfies a commutativity
condition. -/
@[simps]
def homEquiv :
    (PushoutI φ →* K) ≃ { f : (Π i, G i →* K) × (H →* K) // ∀ i, (f.1 i).comp (φ i) = f.2 } :=
  { toFun := fun f => ⟨(fun i => f.comp (of i), f.comp (base φ)),
                  /-
                    ι : Type u_1
                    G : ι → Type u_2
                    H : Type u_3
                    K : Type u_4
                    inst✝² : Monoid K
                    inst✝¹ : (i : ι) → Monoid (G i)
                    inst✝ : Monoid H
                    φ : (i : ι) → MonoidHom H (G i)
                    f : MonoidHom (Monoid.PushoutI φ) K
                    i : ι
                    ⊢ Eq (({ fst := fun i => f.comp (Monoid.PushoutI.of i), snd := f.comp (Monoid. …
                  -/
      fun i => by rw [MonoidHom.comp_assoc, of_comp_eq_base]⟩
                  /-
                    🎉 no goals
                  -/
    invFun := fun f => lift f.1.1 f.1.2 f.2,
                                     /-
                                       ι : Type u_1
                                       G : ι → Type u_2
                                       H : Type u_3
                                       K : Type u_4
                                       inst✝² : Monoid K
                                       inst✝¹ : (i : ι) → Monoid (G i)
                                       inst✝ : Monoid H
                                       φ : (i : ι) → MonoidHom H (G i)
                                       x✝ : MonoidHom (Monoid.PushoutI φ) K
                                       ⊢ ∀ (i : ι), Eq (((fun f => Monoid.PushoutI.lift (↑f).1 (↑f).2 ⋯) ((fun f => ⟨ …
                                     -/
    left_inv := fun _ => hom_ext (by simp [DFunLike.ext_iff])
                                     /-
                                       🎉 no goals
                                     -/
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝² : Monoid K
            inst✝¹ : (i : ι) → Monoid (G i)
            inst✝ : Monoid H
            φ : (i : ι) → MonoidHom H (G i)
            x✝ : MonoidHom (Monoid.PushoutI φ) K
            ⊢ Eq (((fun f => Monoid.PushoutI.lift (↑f).1 (↑f).2 ⋯) ((fun f => ⟨{ fst := fu …
          -/
      (by simp [DFunLike.ext_iff])
          /-
            🎉 no goals
          -/
                                       /-
                                         ι : Type u_1
                                         G : ι → Type u_2
                                         H : Type u_3
                                         K : Type u_4
                                         inst✝² : Monoid K
                                         inst✝¹ : (i : ι) → Monoid (G i)
                                         inst✝ : Monoid H
                                         φ : (i : ι) → MonoidHom H (G i)
                                         x✝ : Subtype fun f => ∀ (i : ι), Eq ((f.1 i).comp (φ i)) f.2
                                         fst✝ : (i : ι) → MonoidHom (G i) K
                                         snd✝ : MonoidHom H K
                                         property✝ : ∀ (i : ι), Eq (({ fst := fst✝, snd := snd✝ }.1 i).comp (φ i)) { fs …
                                         ⊢ Eq ((fun f => ⟨{ fst := fun i => f.comp (Monoid.PushoutI.of i), snd := f.com …
                                       -/
    right_inv := fun ⟨⟨_, _⟩, _⟩ => by simp [DFunLike.ext_iff, funext_iff] }
                                       /-
                                         🎉 no goals
                                       -/


/-- The map from the coproduct into the pushout -/
def ofCoprodI : CoprodI G →* PushoutI φ :=
  CoprodI.lift of


@[simp]
theorem ofCoprodI_of (i : ι) (g : G i) :
    (ofCoprodI (CoprodI.of g) : PushoutI φ) = of i g := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    i : ι
    g : G i
    ⊢ Eq (Monoid.PushoutI.ofCoprodI (Monoid.CoprodI.of g)) ((Monoid.PushoutI.of i) …
  -/
  simp [ofCoprodI]
  /-
    🎉 no goals
  -/


theorem induction_on {motive : PushoutI φ → Prop}
    (x : PushoutI φ)
    (of : ∀ (i : ι) (g : G i), motive (of i g))
    (base : ∀ h, motive (base φ h))
    (mul : ∀ x y, motive x → motive y → motive (x * y)) : motive x := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Monoid (G i)
    inst✝ : Monoid H
    φ : (i : ι) → MonoidHom H (G i)
    motive : Monoid.PushoutI φ → Prop
    x : Monoid.PushoutI φ
    of : ∀ (i : ι) (g : G i), motive ((Monoid.PushoutI.of i) g)
    base : ∀ (h : H), motive ((Monoid.PushoutI.base φ) h)
    mul : ∀ (x y : Monoid.PushoutI φ), motive x → motive y → motive (HMul.hMul x y)
    ⊢ motive x
  -/
  delta PushoutI PushoutI.of PushoutI.base at *
  induction x using Con.induction_on with
  | H x =>
    induction x using Coprod.induction_on with
    | inl g =>
      induction g using CoprodI.induction_on with
      | h_of i g => exact of i g
      | h_mul x y ihx ihy =>
        rw [map_mul]
        exact mul _ _ ihx ihy
      | h_one => simpa using base 1
    | inr h => exact base h
    | mul x y ihx ihy => exact mul _ _ ihx ihy


instance : Group (PushoutI φ) :=
  { Con.group (PushoutI.con φ) with
    toMonoid := PushoutI.monoid }


/-- The data we need to pick a normal form for words in the pushout. We need to pick a
canonical element of each coset. We also need all the maps in the diagram to be injective  -/
structure Transversal : Type _ where
  /-- All maps in the diagram are injective -/
  injective : ∀ i, Injective (φ i)
  /-- The underlying set, containing exactly one element of each coset of the base group -/
  set : ∀ i, Set (G i)
  /-- The chosen element of the base group itself is the identity -/
  one_mem : ∀ i, 1 ∈ set i
  /-- We have exactly one element of each coset of the base group -/
  compl : ∀ i, IsComplement (φ i).range (set i)


theorem transversal_nonempty (hφ : ∀ i, Injective (φ i)) : Nonempty (Transversal φ) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    ⊢ Nonempty (Monoid.PushoutI.NormalWord.Transversal φ)
  -/
  choose t ht using fun i => (φ i).range.exists_isComplement_right 1
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    t : (i : ι) → Set (G i)
    ht : ∀ (i : ι), And (Subgroup.IsComplement (↑(φ i).range) (t i)) (Membership.m …
    ⊢ Nonempty (Monoid.PushoutI.NormalWord.Transversal φ)
  -/
  apply Nonempty.intro
  exact
    { injective := hφ
      set := t
      one_mem := fun i => (ht i).2
      compl := fun i => (ht i).1 }


/-- The normal form for words in the pushout. Every element of the pushout is the product of an
element of the base group and a word made up of letters each of which is in the transversal. -/
structure _root_.Monoid.PushoutI.NormalWord (d : Transversal φ) extends CoprodI.Word G where
  /-- Every `NormalWord` is the product of an element of the base group and a word made up
  of letters each of which is in the transversal. `head` is that element of the base group. -/
  head : H
  /-- All letter in the word are in the transversal. -/
  normalized : ∀ i g, ⟨i, g⟩ ∈ toList → g ∈ d.set i


/--
A `Pair d i` is a word in the coproduct, `Coprod G`, the `tail`, and an element of the group `G i`,
the `head`. The first letter of the `tail` must not be an element of `G i`.
Note that the `head` may be `1` Every letter in the `tail` must be in the transversal given by `d`.
Similar to `Monoid.CoprodI.Pair` except every letter must be in the transversal
(not including the head letter). -/
structure Pair (d : Transversal φ) (i : ι) extends CoprodI.Word.Pair G i where
  /-- All letters in the word are in the transversal. -/
  normalized : ∀ i g, ⟨i, g⟩ ∈ tail.toList → g ∈ d.set i


/-- The empty normalized word, representing the identity element of the group. -/
@[simps!]
                                                                  /-
                                                                    ι : Type u_1
                                                                    G : ι → Type u_2
                                                                    H : Type u_3
                                                                    K : Type u_4
                                                                    inst✝² : Monoid K
                                                                    inst✝¹ : (i : ι) → Group (G i)
                                                                    inst✝ : Group H
                                                                    φ : (i : ι) → MonoidHom H (G i)
                                                                    d : Monoid.PushoutI.NormalWord.Transversal φ
                                                                    i : ι
                                                                    g : G i
                                                                    ⊢ Membership.mem Monoid.CoprodI.Word.empty.toList ⟨i, g⟩ → Membership.mem (d.s …
                                                                  -/
def empty : NormalWord d := ⟨CoprodI.Word.empty, 1, fun i g => by simp [CoprodI.Word.empty]⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance : Inhabited (NormalWord d) := ⟨NormalWord.empty⟩


instance (i : ι) : Inhabited (Pair d i) :=
  ⟨{ (empty : NormalWord d) with
      head := 1,
                               /-
                                 ι : Type u_1
                                 G : ι → Type u_2
                                 H : Type u_3
                                 K : Type u_4
                                 inst✝² : Monoid K
                                 inst✝¹ : (i : ι) → Group (G i)
                                 inst✝ : Group H
                                 φ : (i : ι) → MonoidHom H (G i)
                                 d : Monoid.PushoutI.NormalWord.Transversal φ
                                 i : ι
                                 h : Eq __src✝.1.fstIdx (Option.some i)
                                 ⊢ False
                               -/
      fstIdx_ne := fun h => by cases h }⟩
                               /-
                                 🎉 no goals
                               -/


@[ext]
theorem ext {w₁ w₂ : NormalWord d} (hhead : w₁.head = w₂.head)
    (hlist : w₁.toList = w₂.toList) : w₁ = w₂ := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    w₁ w₂ : Monoid.PushoutI.NormalWord d
    hhead : Eq w₁.head w₂.head
    hlist : Eq w₁.toList w₂.toList
    ⊢ Eq w₁ w₂
  -/
  rcases w₁ with ⟨⟨_, _, _⟩, _, _⟩
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    w₂ : Monoid.PushoutI.NormalWord d
    head✝ : H
    toList✝ : List (Sigma fun i => G i)
    ne_one✝ : ∀ (l : Sigma fun i => G i), Membership.mem toList✝ l → Ne l.snd 1
    chain_ne✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) toList✝
    normalized✝ : ∀ (i : ι) (g : G i), Membership.mem { toList := toList✝, ne_one  …
    hhead : Eq { toList := toList✝, ne_one := ne_one✝, chain_ne := chain_ne✝, head …
    hlist : Eq { toList := toList✝, ne_one := ne_one✝, chain_ne := chain_ne✝, head …
    ⊢ Eq { toList := toList✝, ne_one := ne_one✝, chain_ne := chain_ne✝, head := he …
  -/
  rcases w₂ with ⟨⟨_, _, _⟩, _, _⟩
  /-
    case mk.mk.mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    head✝¹ : H
    toList✝¹ : List (Sigma fun i => G i)
    ne_one✝¹ : ∀ (l : Sigma fun i => G i), Membership.mem toList✝¹ l → Ne l.snd 1
    chain_ne✝¹ : List.Chain' (fun l l' => Ne l.fst l'.fst) toList✝¹
    normalized✝¹ : ∀ (i : ι) (g : G i), Membership.mem { toList := toList✝¹, ne_on …
    head✝ : H
    toList✝ : List (Sigma fun i => G i)
    ne_one✝ : ∀ (l : Sigma fun i => G i), Membership.mem toList✝ l → Ne l.snd 1
    chain_ne✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) toList✝
    normalized✝ : ∀ (i : ι) (g : G i), Membership.mem { toList := toList✝, ne_one  …
    hhead : Eq { toList := toList✝¹, ne_one := ne_one✝¹, chain_ne := chain_ne✝¹, h …
    hlist : Eq { toList := toList✝¹, ne_one := ne_one✝¹, chain_ne := chain_ne✝¹, h …
    ⊢ Eq { toList := toList✝¹, ne_one := ne_one✝¹, chain_ne := chain_ne✝¹, head := …
  -/
  simp_all
  /-
    🎉 no goals
  -/


instance baseAction : MulAction H (NormalWord d) :=
  { smul := fun h w => { w with head := h * w.head },
                   /-
                     ι : Type u_1
                     G : ι → Type u_2
                     H : Type u_3
                     K : Type u_4
                     inst✝² : Monoid K
                     inst✝¹ : (i : ι) → Group (G i)
                     inst✝ : Group H
                     φ : (i : ι) → MonoidHom H (G i)
                     d : Monoid.PushoutI.NormalWord.Transversal φ
                     ⊢ ∀ (b : Monoid.PushoutI.NormalWord d), Eq (HSMul.hSMul 1 b) b
                   -/
    one_smul := by simp [instHSMul]
                   /-
                     🎉 no goals
                   -/
                   /-
                     ι : Type u_1
                     G : ι → Type u_2
                     H : Type u_3
                     K : Type u_4
                     inst✝² : Monoid K
                     inst✝¹ : (i : ι) → Group (G i)
                     inst✝ : Group H
                     φ : (i : ι) → MonoidHom H (G i)
                     d : Monoid.PushoutI.NormalWord.Transversal φ
                     ⊢ ∀ (x y : H) (b : Monoid.PushoutI.NormalWord d), Eq (HSMul.hSMul (HMul.hMul x …
                   -/
    mul_smul := by simp [instHSMul, mul_assoc] }
                   /-
                     🎉 no goals
                   -/


theorem base_smul_def' (h : H) (w : NormalWord d) :
    h • w = { w with head := h * w.head } := rfl

/-- Take the product of a normal word as an element of the `PushoutI`. We show that this is
bijective, in `NormalWord.equiv`. -/
def prod (w : NormalWord d) : PushoutI φ :=
  base φ w.head * ofCoprodI (w.toWord).prod


@[simp]
theorem prod_base_smul (h : H) (w : NormalWord d) :
    (h • w).prod = base φ h * w.prod := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    h : H
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (HSMul.hSMul h w).prod (HMul.hMul ((Monoid.PushoutI.base φ) h) w.prod)
  -/
  simp only [base_smul_def', prod, map_mul, mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_empty : (empty : NormalWord d).prod = 1 := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    ⊢ Eq Monoid.PushoutI.NormalWord.empty.prod 1
  -/
  simp [prod, empty]
  /-
    🎉 no goals
  -/


/-- A constructor that multiplies a `NormalWord` by an element, with condition to make
sure the underlying list does get longer. -/
@[simps!]
noncomputable def cons {i} (g : G i) (w : NormalWord d) (hmw : w.fstIdx ≠ some i)
    (hgr : g ∉ (φ i).range) : NormalWord d :=
  letI n := (d.compl i).equiv (g * (φ i w.head))
  letI w' := Word.cons (n.2 : G i) w.toWord hmw
    (mt (coe_equiv_snd_eq_one_iff_mem _ (d.one_mem _)).1
                                    /-
                                      ι : Type u_1
                                      G : ι → Type u_2
                                      H : Type u_3
                                      K : Type u_4
                                      inst✝² : Monoid K
                                      inst✝¹ : (i : ι) → Group (G i)
                                      inst✝ : Group H
                                      φ : (i : ι) → MonoidHom H (G i)
                                      d : Monoid.PushoutI.NormalWord.Transversal φ
                                      i : ι
                                      g : G i
                                      w : Monoid.PushoutI.NormalWord d
                                      hmw : Ne w.fstIdx (Option.some i)
                                      hgr : Not (Membership.mem (φ i).range g)
                                      n : Prod ↑↑(φ i).range ↑(d.set i) := ⋯.equiv (HMul.hMul g ((φ i) w.head))
                                      ⊢ Membership.mem (φ i).range ((φ i) w.head)
                                    -/
      (mt (mul_mem_cancel_right (by simp)).1 hgr))
                                    /-
                                      🎉 no goals
                                    -/
  { toWord := w'
    head := (MonoidHom.ofInjective (d.injective i)).symm n.1
    normalized := fun i g hg => by
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝² : Monoid K
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        i✝ : ι
        g✝ : G i✝
        w : Monoid.PushoutI.NormalWord d
        hmw : Ne w.fstIdx (Option.some i✝)
        hgr : Not (Membership.mem (φ i✝).range g✝)
        n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv (HMul.hMul g✝ ((φ i✝) w.head))
        w' : Monoid.CoprodI.Word G := Monoid.CoprodI.Word.cons (↑n.2) w.toWord hmw ⋯
        i : ι
        g : G i
        hg : Membership.mem w'.toList ⟨i, g⟩
        ⊢ Membership.mem (d.set i) g
      -/
      simp only [w', Word.cons, mem_cons, Sigma.mk.inj_iff] at hg
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝² : Monoid K
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        i✝ : ι
        g✝ : G i✝
        w : Monoid.PushoutI.NormalWord d
        hmw : Ne w.fstIdx (Option.some i✝)
        hgr : Not (Membership.mem (φ i✝).range g✝)
        n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv (HMul.hMul g✝ ((φ i✝) w.head))
        w' : Monoid.CoprodI.Word G := Monoid.CoprodI.Word.cons (↑n.2) w.toWord hmw ⋯
        i : ι
        g : G i
        hg : Or (And (Eq i i✝) (HEq g ↑n.2)) (Membership.mem w.toList ⟨i, g⟩)
        ⊢ Membership.mem (d.set i) g
      -/
      rcases hg with ⟨rfl, hg | hg⟩
        /-
          case inl.intro.refl
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          K : Type u_4
          inst✝² : Monoid K
          inst✝¹ : (i : ι) → Group (G i)
          inst✝ : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          w : Monoid.PushoutI.NormalWord d
          i : ι
          g : G i
          hmw : Ne w.fstIdx (Option.some i)
          hgr : Not (Membership.mem (φ i).range g)
          n : Prod ↑↑(φ i).range ↑(d.set i) := ⋯.equiv (HMul.hMul g ((φ i) w.head))
          w' : Monoid.CoprodI.Word G := Monoid.CoprodI.Word.cons (↑n.2) w.toWord hmw ⋯
          β : Type u_2
          ⊢ Membership.mem (d.set i) ↑n.2
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case inr
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          K : Type u_4
          inst✝² : Monoid K
          inst✝¹ : (i : ι) → Group (G i)
          inst✝ : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          i✝ : ι
          g✝ : G i✝
          w : Monoid.PushoutI.NormalWord d
          hmw : Ne w.fstIdx (Option.some i✝)
          hgr : Not (Membership.mem (φ i✝).range g✝)
          n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv (HMul.hMul g✝ ((φ i✝) w.head))
          w' : Monoid.CoprodI.Word G := Monoid.CoprodI.Word.cons (↑n.2) w.toWord hmw ⋯
          i : ι
          g : G i
          h✝ : Membership.mem w.toList ⟨i, g⟩
          ⊢ Membership.mem (d.set i) g
        -/
      · exact w.normalized _ _ (by assumption) }
        /-
          🎉 no goals
        -/


@[simp]
theorem prod_cons {i} (g : G i) (w : NormalWord d) (hmw : w.fstIdx ≠ some i)
    (hgr : g ∉ (φ i).range) : (cons g w hmw hgr).prod = of i g * w.prod := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    i : ι
    g : G i
    w : Monoid.PushoutI.NormalWord d
    hmw : Ne w.fstIdx (Option.some i)
    hgr : Not (Membership.mem (φ i).range g)
    ⊢ Eq (Monoid.PushoutI.NormalWord.cons g w hmw hgr).prod (HMul.hMul ((Monoid.Pu …
  -/
  simp [prod, cons, ← of_apply_eq_base φ i, equiv_fst_eq_mul_inv, mul_assoc]
  /-
    🎉 no goals
  -/


/-- Given a word in `CoprodI`, if every letter is in the transversal and when
we multiply by an element of the base group it still has this property,
then the element of the base group we multiplied by was one. -/
theorem eq_one_of_smul_normalized (w : CoprodI.Word G) {i : ι} (h : H)
    (hw : ∀ i g, ⟨i, g⟩ ∈ w.toList → g ∈ d.set i)
    (hφw : ∀ j g, ⟨j, g⟩ ∈ (CoprodI.of (φ i h) • w).toList → g ∈ d.set j) :
    h = 1 := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    w : Monoid.CoprodI.Word G
    i : ι
    h : H
    hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Membership.mem (d.s …
    hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
    ⊢ Eq h 1
  -/
  simp only [← (d.compl _).equiv_snd_eq_self_iff_mem (one_mem _)] at hw hφw
  have hhead : ((d.compl i).equiv (Word.equivPair i w).head).2 =
      (Word.equivPair i w).head := by
    rw [Word.equivPair_head]
    split_ifs with h
    · rcases h with ⟨_, rfl⟩
      exact hw _ _ (List.head_mem _)
    · rw [equiv_one (d.compl i) (one_mem _) (d.one_mem _)]
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    w : Monoid.CoprodI.Word G
    i : ι
    h : H
    hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
    hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
    hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
    ⊢ Eq h 1
  -/
  by_contra hh1
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    w : Monoid.CoprodI.Word G
    i : ι
    h : H
    hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
    hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
    hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
    hh1 : Not (Eq h 1)
    ⊢ False
  -/
  have := hφw i (φ i h * (Word.equivPair i w).head) ?_
    /-
      case refine_2
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      w : Monoid.CoprodI.Word G
      i : ι
      h : H
      hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
      hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
      hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
      hh1 : Not (Eq h 1)
      this : Eq (↑(⋯.equiv (HMul.hMul ((φ i) h) ((Monoid.CoprodI.Word.equivPair i) w …
      ⊢ False
    -/
  · apply hh1
    /-
      case refine_2
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      w : Monoid.CoprodI.Word G
      i : ι
      h : H
      hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
      hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
      hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
      hh1 : Not (Eq h 1)
      this : Eq (↑(⋯.equiv (HMul.hMul ((φ i) h) ((Monoid.CoprodI.Word.equivPair i) w …
      ⊢ Eq h 1
    -/
    rw [equiv_mul_left_of_mem (d.compl i) ⟨_, rfl⟩, hhead] at this
    /-
      case refine_2
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      w : Monoid.CoprodI.Word G
      i : ι
      h : H
      hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
      hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
      hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
      hh1 : Not (Eq h 1)
      this : Eq ((Monoid.CoprodI.Word.equivPair i) w).head (HMul.hMul ((φ i) h) ((Mo …
      ⊢ Eq h 1
    -/
    simpa [((injective_iff_map_eq_one' _).1 (d.injective i))] using this
    /-
      🎉 no goals
    -/
  · simp only [Word.mem_smul_iff, not_true, false_and, ne_eq, Option.mem_def, mul_right_inj,
      exists_eq_right', mul_right_eq_self, exists_prop, true_and, false_or]
    /-
      case refine_1
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      w : Monoid.CoprodI.Word G
      i : ι
      h : H
      hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
      hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
      hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
      hh1 : Not (Eq h 1)
      ⊢ And (Not (Eq (HMul.hMul ((φ i) h) ((Monoid.CoprodI.Word.equivPair i) w).head …
    -/
    constructor
      /-
        case refine_1.left
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        ⊢ Not (Eq (HMul.hMul ((φ i) h) ((Monoid.CoprodI.Word.equivPair i) w).head) 1)
      -/
    · intro h
      /-
        case refine_1.left
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h✝ : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h✝ 1)
        h : Eq (HMul.hMul ((φ i) h✝) ((Monoid.CoprodI.Word.equivPair i) w).head) 1
        ⊢ False
      -/
      apply_fun (d.compl i).equiv at h
      simp only [Prod.ext_iff, equiv_one (d.compl i) (one_mem _) (d.one_mem _),
        equiv_mul_left_of_mem (d.compl i) ⟨_, rfl⟩ , hhead, Subtype.ext_iff,
        Prod.ext_iff, Subgroup.coe_mul] at h
      /-
        case refine_1.left
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h✝ : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h✝ 1)
        h : And (Eq (↑(HMul.hMul ⟨(φ i) h✝, ⋯⟩ (⋯.equiv ((Monoid.CoprodI.Word.equivPai …
        ⊢ False
      -/
      rcases h with ⟨h₁, h₂⟩
      /-
        case refine_1.left.intro
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        h₁ : Eq (↑(HMul.hMul ⟨(φ i) h, ⋯⟩ (⋯.equiv ((Monoid.CoprodI.Word.equivPair i)  …
        h₂ : Eq ((Monoid.CoprodI.Word.equivPair i) w).head 1
        ⊢ False
      -/
      rw [h₂, equiv_one (d.compl i) (one_mem _) (d.one_mem _)] at h₁
      /-
        case refine_1.left.intro
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        h₁ : Eq (↑(HMul.hMul ⟨(φ i) h, ⋯⟩ { fst := ⟨1, ⋯⟩, snd := ⟨1, ⋯⟩ }.1)) 1
        h₂ : Eq ((Monoid.CoprodI.Word.equivPair i) w).head 1
        ⊢ False
      -/
      erw [mul_one] at h₁
      /-
        case refine_1.left.intro
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        h₁ : Eq (↑⟨(φ i) h, ⋯⟩) 1
        h₂ : Eq ((Monoid.CoprodI.Word.equivPair i) w).head 1
        ⊢ False
      -/
      simp only [((injective_iff_map_eq_one' _).1 (d.injective i))] at h₁
      /-
        case refine_1.left.intro
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        h₂ : Eq ((Monoid.CoprodI.Word.equivPair i) w).head 1
        h₁ : Eq h 1
        ⊢ False
      -/
      contradiction
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) ((Monoid.CoprodI.Wo …
      -/
    · rw [Word.equivPair_head]
      /-
        case refine_1.right
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) (dite (Exists fun h …
      -/
      dsimp
      /-
        case refine_1.right
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        w : Monoid.CoprodI.Word G
        i : ι
        h : H
        hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
        hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
        hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
        hh1 : Not (Eq h 1)
        ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) (dite (Exists fun h …
      -/
      split_ifs with hep
        /-
          case pos
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          w : Monoid.CoprodI.Word G
          i : ι
          h : H
          hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
          hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
          hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
          hh1 : Not (Eq h 1)
          hep : Exists fun h => Eq (w.toList.head h).fst i
          ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) (Eq.rec (w.toList.h …
        -/
      · rcases hep with ⟨hnil, rfl⟩
        /-
          case pos.intro
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          w : Monoid.CoprodI.Word G
          h : H
          hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
          hh1 : Not (Eq h 1)
          hnil : Not (Eq w.toList List.nil)
          hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
          hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair (w.toList.head hnil).fst …
          ⊢ Or (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, HMul.hMul ((φ (w …
        -/
        rw [head?_eq_head hnil]
        /-
          case pos.intro
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          w : Monoid.CoprodI.Word G
          h : H
          hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
          hh1 : Not (Eq h 1)
          hnil : Not (Eq w.toList List.nil)
          hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
          hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair (w.toList.head hnil).fst …
          ⊢ Or (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, HMul.hMul ((φ (w …
        -/
        simp_all
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          w : Monoid.CoprodI.Word G
          i : ι
          h : H
          hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
          hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
          hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
          hh1 : Not (Eq h 1)
          hep : Not (Exists fun h => Eq (w.toList.head h).fst i)
          ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) 1⟩) (Or (Eq w.toLis …
        -/
      · push_neg at hep
        /-
          case neg
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          w : Monoid.CoprodI.Word G
          i : ι
          h : H
          hw : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2) g
          hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
          hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
          hh1 : Not (Eq h 1)
          hep : ∀ (h : Not (Eq w.toList List.nil)), Ne (w.toList.head h).fst i
          ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) 1⟩) (Or (Eq w.toLis …
        -/
        by_cases hw : w.toList = []
          /-
            case pos
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            w : Monoid.CoprodI.Word G
            i : ι
            h : H
            hw✝ : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2 …
            hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
            hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
            hh1 : Not (Eq h 1)
            hep : ∀ (h : Not (Eq w.toList List.nil)), Ne (w.toList.head h).fst i
            hw : Eq w.toList List.nil
            ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) 1⟩) (Or (Eq w.toLis …
          -/
        · simp [hw, Word.fstIdx]
          /-
            🎉 no goals
          -/
          /-
            case neg
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            w : Monoid.CoprodI.Word G
            i : ι
            h : H
            hw✝ : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Eq (↑(⋯.equiv g).2 …
            hφw : ∀ (j : ι) (g : G j), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of ((φ  …
            hhead : Eq (↑(⋯.equiv ((Monoid.CoprodI.Word.equivPair i) w).head).2) ((Monoid. …
            hh1 : Not (Eq h 1)
            hep : ∀ (h : Not (Eq w.toList List.nil)), Ne (w.toList.head h).fst i
            hw : Not (Eq w.toList List.nil)
            ⊢ Or (Membership.mem w.toList.tail ⟨i, HMul.hMul ((φ i) h) 1⟩) (Or (Eq w.toLis …
          -/
        · simp [head?_eq_head hw, Word.fstIdx, hep hw]
          /-
            🎉 no goals
          -/


theorem ext_smul {w₁ w₂ : NormalWord d} (i : ι)
    (h : CoprodI.of (φ i w₁.head) • w₁.toWord =
         CoprodI.of (φ i w₂.head) • w₂.toWord) :
    w₁ = w₂ := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    w₁ w₂ : Monoid.PushoutI.NormalWord d
    i : ι
    h : Eq (HSMul.hSMul (Monoid.CoprodI.of ((φ i) w₁.head)) w₁.toWord) (HSMul.hSMu …
    ⊢ Eq w₁ w₂
  -/
  rcases w₁ with ⟨w₁, h₁, hw₁⟩
  /-
    case mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    w₂ : Monoid.PushoutI.NormalWord d
    i : ι
    w₁ : Monoid.CoprodI.Word G
    h₁ : H
    hw₁ : ∀ (i : ι) (g : G i), Membership.mem w₁.toList ⟨i, g⟩ → Membership.mem (d …
    h : Eq (HSMul.hSMul (Monoid.CoprodI.of ((φ i) { toWord := w₁, head := h₁, norm …
    ⊢ Eq { toWord := w₁, head := h₁, normalized := hw₁ } w₂
  -/
  rcases w₂ with ⟨w₂, h₂, hw₂⟩
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    w₁ : Monoid.CoprodI.Word G
    h₁ : H
    hw₁ : ∀ (i : ι) (g : G i), Membership.mem w₁.toList ⟨i, g⟩ → Membership.mem (d …
    w₂ : Monoid.CoprodI.Word G
    h₂ : H
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    h : Eq (HSMul.hSMul (Monoid.CoprodI.of ((φ i) { toWord := w₁, head := h₁, norm …
    ⊢ Eq { toWord := w₁, head := h₁, normalized := hw₁ } { toWord := w₂, head := h …
  -/
  dsimp at *
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    w₁ : Monoid.CoprodI.Word G
    h₁ : H
    hw₁ : ∀ (i : ι) (g : G i), Membership.mem w₁.toList ⟨i, g⟩ → Membership.mem (d …
    w₂ : Monoid.CoprodI.Word G
    h₂ : H
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    h : Eq (HSMul.hSMul (Monoid.CoprodI.of ((φ i) h₁)) w₁) (HSMul.hSMul (Monoid.Co …
    ⊢ Eq { toWord := w₁, head := h₁, normalized := hw₁ } { toWord := w₂, head := h …
  -/
  rw [smul_eq_iff_eq_inv_smul, ← mul_smul] at h
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    w₁ : Monoid.CoprodI.Word G
    h₁ : H
    hw₁ : ∀ (i : ι) (g : G i), Membership.mem w₁.toList ⟨i, g⟩ → Membership.mem (d …
    w₂ : Monoid.CoprodI.Word G
    h₂ : H
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    h : Eq w₁ (HSMul.hSMul (HMul.hMul (Inv.inv (Monoid.CoprodI.of ((φ i) h₁))) (Mo …
    ⊢ Eq { toWord := w₁, head := h₁, normalized := hw₁ } { toWord := w₂, head := h …
  -/
  subst h
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    h₁ : H
    w₂ : Monoid.CoprodI.Word G
    h₂ : H
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    hw₁ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv …
    ⊢ Eq { toWord := HSMul.hSMul (HMul.hMul (Inv.inv (Monoid.CoprodI.of ((φ i) h₁) …
  -/
  simp only [← map_inv, ← map_mul] at hw₁
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    h₁ : H
    w₂ : Monoid.CoprodI.Word G
    h₂ : H
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    hw₁✝ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (HMul.hMul (Inv.in …
    hw₁ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of  …
    ⊢ Eq { toWord := HSMul.hSMul (HMul.hMul (Inv.inv (Monoid.CoprodI.of ((φ i) h₁) …
  -/
  have : h₁⁻¹ * h₂ = 1 := eq_one_of_smul_normalized w₂ (h₁⁻¹ * h₂) hw₂ hw₁
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    h₁ : H
    w₂ : Monoid.CoprodI.Word G
    h₂ : H
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    hw₁✝ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (HMul.hMul (Inv.in …
    hw₁ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of  …
    this : Eq (HMul.hMul (Inv.inv h₁) h₂) 1
    ⊢ Eq { toWord := HSMul.hSMul (HMul.hMul (Inv.inv (Monoid.CoprodI.of ((φ i) h₁) …
  -/
  rw [inv_mul_eq_one] at this; subst this
  /-
    case mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    h₁ : H
    w₂ : Monoid.CoprodI.Word G
    hw₂ : ∀ (i : ι) (g : G i), Membership.mem w₂.toList ⟨i, g⟩ → Membership.mem (d …
    hw₁✝ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (HMul.hMul (Inv.in …
    hw₁ : ∀ (i_1 : ι) (g : G i_1), Membership.mem (HSMul.hSMul (Monoid.CoprodI.of  …
    ⊢ Eq { toWord := HSMul.hSMul (HMul.hMul (Inv.inv (Monoid.CoprodI.of ((φ i) h₁) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a pair `(head, tail)`, we can form a word by prepending `head` to `tail`, but
putting head into normal form first, by making sure it is expressed as an element
of the base group multiplied by an element of the transversal. -/
noncomputable def rcons (i : ι) (p : Pair d i) : NormalWord d :=
  letI n := (d.compl i).equiv p.head
  let w := (Word.equivPair i).symm { p.toPair with head := n.2 }
  { toWord := w
    head := (MonoidHom.ofInjective (d.injective i)).symm n.1
    normalized := fun i g hg => by
        /-
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          K : Type u_4
          inst✝⁴ : Monoid K
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          i✝ : ι
          p : Monoid.PushoutI.NormalWord.Pair d i✝
          n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv p.head
          w : Monoid.CoprodI.Word G :=
            (Monoid.CoprodI.Word.equivPair i✝).symm
              (let __src := p.toPair;
              { head := ↑n.2, tail := __src.tail, fstIdx_ne := ⋯ })
          i : ι
          g : G i
          hg : Membership.mem w.toList ⟨i, g⟩
          ⊢ Membership.mem (d.set i) g
        -/
        dsimp [w] at hg
        /-
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          K : Type u_4
          inst✝⁴ : Monoid K
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          i✝ : ι
          p : Monoid.PushoutI.NormalWord.Pair d i✝
          n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv p.head
          w : Monoid.CoprodI.Word G :=
            (Monoid.CoprodI.Word.equivPair i✝).symm
              (let __src := p.toPair;
              { head := ↑n.2, tail := __src.tail, fstIdx_ne := ⋯ })
          i : ι
          g : G i
          hg : Membership.mem ((Monoid.CoprodI.Word.equivPair i✝).symm { head := ↑n.2, t …
          ⊢ Membership.mem (d.set i) g
        -/
        rw [Word.equivPair_symm, Word.mem_rcons_iff] at hg
        /-
          ι : Type u_1
          G : ι → Type u_2
          H : Type u_3
          K : Type u_4
          inst✝⁴ : Monoid K
          inst✝³ : (i : ι) → Group (G i)
          inst✝² : Group H
          φ : (i : ι) → MonoidHom H (G i)
          d : Monoid.PushoutI.NormalWord.Transversal φ
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (G i)
          i✝ : ι
          p : Monoid.PushoutI.NormalWord.Pair d i✝
          n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv p.head
          w : Monoid.CoprodI.Word G :=
            (Monoid.CoprodI.Word.equivPair i✝).symm
              (let __src := p.toPair;
              { head := ↑n.2, tail := __src.tail, fstIdx_ne := ⋯ })
          i : ι
          g : G i
          hg : Or (Membership.mem { head := ↑n.2, tail := p.tail, fstIdx_ne := ⋯ }.tail. …
          ⊢ Membership.mem (d.set i) g
        -/
        rcases hg with hg | ⟨_, rfl, rfl⟩
          /-
            case inl
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝⁴ : Monoid K
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            i✝ : ι
            p : Monoid.PushoutI.NormalWord.Pair d i✝
            n : Prod ↑↑(φ i✝).range ↑(d.set i✝) := ⋯.equiv p.head
            w : Monoid.CoprodI.Word G :=
              (Monoid.CoprodI.Word.equivPair i✝).symm
                (let __src := p.toPair;
                { head := ↑n.2, tail := __src.tail, fstIdx_ne := ⋯ })
            i : ι
            g : G i
            hg : Membership.mem { head := ↑n.2, tail := p.tail, fstIdx_ne := ⋯ }.tail.toLi …
            ⊢ Membership.mem (d.set i) g
          -/
        · exact p.normalized _ _ hg
          /-
            🎉 no goals
          -/
          /-
            case inr.intro.intro
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝⁴ : Monoid K
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            i : ι
            p : Monoid.PushoutI.NormalWord.Pair d i
            n : Prod ↑↑(φ i).range ↑(d.set i) := ⋯.equiv p.head
            w : Monoid.CoprodI.Word G :=
              (Monoid.CoprodI.Word.equivPair i).symm
                (let __src := p.toPair;
                { head := ↑n.2, tail := __src.tail, fstIdx_ne := ⋯ })
            left✝ : Ne (Eq.rec { head := ↑n.2, tail := p.tail, fstIdx_ne := ⋯ }.head ⋯) 1
            ⊢ Membership.mem (d.set i) (Eq.rec { head := ↑n.2, tail := p.tail, fstIdx_ne : …
          -/
        · simp }
          /-
            🎉 no goals
          -/


theorem rcons_injective {i : ι} : Function.Injective (rcons (d := d) i) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    ⊢ Function.Injective (Monoid.PushoutI.NormalWord.rcons i)
  -/
  rintro ⟨⟨head₁, tail₁⟩, _⟩ ⟨⟨head₂, tail₂⟩, _⟩
  simp only [rcons, NormalWord.mk.injEq, EmbeddingLike.apply_eq_iff_eq,
    Word.Pair.mk.injEq, Pair.mk.injEq, and_imp]
  /-
    case mk.mk.mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    head₁ : G i
    tail₁ : Monoid.CoprodI.Word G
    fstIdx_ne✝¹ : Ne tail₁.fstIdx (Option.some i)
    normalized✝¹ : ∀ (i_1 : ι) (g : G i_1), Membership.mem { head := head₁, tail : …
    head₂ : G i
    tail₂ : Monoid.CoprodI.Word G
    fstIdx_ne✝ : Ne tail₂.fstIdx (Option.some i)
    normalized✝ : ∀ (i_1 : ι) (g : G i_1), Membership.mem { head := head₂, tail := …
    ⊢ Eq ↑(⋯.equiv head₁).2 ↑(⋯.equiv head₂).2 → Eq tail₁ tail₂ → Eq (⋯.equiv head …
  -/
  intro h₁ h₂ h₃
  /-
    case mk.mk.mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    head₁ : G i
    tail₁ : Monoid.CoprodI.Word G
    fstIdx_ne✝¹ : Ne tail₁.fstIdx (Option.some i)
    normalized✝¹ : ∀ (i_1 : ι) (g : G i_1), Membership.mem { head := head₁, tail : …
    head₂ : G i
    tail₂ : Monoid.CoprodI.Word G
    fstIdx_ne✝ : Ne tail₂.fstIdx (Option.some i)
    normalized✝ : ∀ (i_1 : ι) (g : G i_1), Membership.mem { head := head₂, tail := …
    h₁ : Eq ↑(⋯.equiv head₁).2 ↑(⋯.equiv head₂).2
    h₂ : Eq tail₁ tail₂
    h₃ : Eq (⋯.equiv head₁).1 (⋯.equiv head₂).1
    ⊢ And (Eq head₁ head₂) (Eq tail₁ tail₂)
  -/
  subst h₂
  rw [← equiv_fst_mul_equiv_snd (d.compl i) head₁,
      ← equiv_fst_mul_equiv_snd (d.compl i) head₂,
    h₁, h₃]
  /-
    case mk.mk.mk.mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    head₁ : G i
    tail₁ : Monoid.CoprodI.Word G
    fstIdx_ne✝¹ : Ne tail₁.fstIdx (Option.some i)
    normalized✝¹ : ∀ (i_1 : ι) (g : G i_1), Membership.mem { head := head₁, tail : …
    head₂ : G i
    h₁ : Eq ↑(⋯.equiv head₁).2 ↑(⋯.equiv head₂).2
    h₃ : Eq (⋯.equiv head₁).1 (⋯.equiv head₂).1
    fstIdx_ne✝ : Ne tail₁.fstIdx (Option.some i)
    normalized✝ : ∀ (i_1 : ι) (g : G i_1), Membership.mem { head := head₂, tail := …
    ⊢ And (Eq (HMul.hMul ↑(⋯.equiv head₂).1 ↑(⋯.equiv head₂).2) (HMul.hMul ↑(⋯.equ …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The equivalence between `NormalWord`s and pairs. We can turn a `NormalWord` into a
pair by taking the head of the `List` if it is in `G i` and multiplying it by the element of the
base group. -/
noncomputable def equivPair (i) : NormalWord d ≃ Pair d i :=
  letI toFun : NormalWord d → Pair d i :=
    fun w =>
      letI p := Word.equivPair i (CoprodI.of (φ i w.head) • w.toWord)
      { toPair := p
        normalized := fun j g hg => by
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝⁴ : Monoid K
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            i : ι
            w : Monoid.PushoutI.NormalWord d
            p : Monoid.CoprodI.Word.Pair G i := (Monoid.CoprodI.Word.equivPair i) (HSMul.h …
            j : ι
            g : G j
            hg : Membership.mem p.tail.toList ⟨j, g⟩
            ⊢ Membership.mem (d.set j) g
          -/
          dsimp only [p] at hg
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝⁴ : Monoid K
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            i : ι
            w : Monoid.PushoutI.NormalWord d
            p : Monoid.CoprodI.Word.Pair G i := (Monoid.CoprodI.Word.equivPair i) (HSMul.h …
            j : ι
            g : G j
            hg : Membership.mem ((Monoid.CoprodI.Word.equivPair i) (HSMul.hSMul (Monoid.Co …
            ⊢ Membership.mem (d.set j) g
          -/
          rw [Word.of_smul_def, ← Word.equivPair_symm, Equiv.apply_symm_apply] at hg
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝⁴ : Monoid K
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            i : ι
            w : Monoid.PushoutI.NormalWord d
            p : Monoid.CoprodI.Word.Pair G i := (Monoid.CoprodI.Word.equivPair i) (HSMul.h …
            j : ι
            g : G j
            hg :
              Membership.mem
                (let __src := (Monoid.CoprodI.Word.equivPair i) w.toWord;
                    { head := HMul.hMul ((φ i) w.head) ((Monoid.CoprodI.Word.equivPair i)  …
                ⟨j, g⟩
            ⊢ Membership.mem (d.set j) g
          -/
          dsimp at hg
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            K : Type u_4
            inst✝⁴ : Monoid K
            inst✝³ : (i : ι) → Group (G i)
            inst✝² : Group H
            φ : (i : ι) → MonoidHom H (G i)
            d : Monoid.PushoutI.NormalWord.Transversal φ
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (G i)
            i : ι
            w : Monoid.PushoutI.NormalWord d
            p : Monoid.CoprodI.Word.Pair G i := (Monoid.CoprodI.Word.equivPair i) (HSMul.h …
            j : ι
            g : G j
            hg : Membership.mem ((Monoid.CoprodI.Word.equivPair i) w.toWord).tail.toList ⟨ …
            ⊢ Membership.mem (d.set j) g
          -/
          exact w.normalized _ _ (Word.mem_of_mem_equivPair_tail _ hg) }
          /-
            🎉 no goals
          -/
  haveI leftInv : Function.LeftInverse (rcons i) toFun :=
    fun w => ext_smul i <| by
      simp only [toFun, rcons, Word.equivPair_symm,
        Word.equivPair_smul_same, Word.equivPair_tail_eq_inv_smul, Word.rcons_eq_smul,
        MonoidHom.apply_ofInjective_symm, equiv_fst_eq_mul_inv, mul_assoc, map_mul, map_inv,
        mul_smul, inv_smul_smul, smul_inv_smul]
  { toFun := toFun
    invFun := rcons i
    left_inv := leftInv
    right_inv := fun _ => rcons_injective (leftInv _) }


noncomputable instance summandAction (i : ι) : MulAction (G i) (NormalWord d) :=
  { smul := fun g w => (equivPair i).symm
      { equivPair i w with
        head := g * (equivPair i w).head }
    one_smul := fun _ => by
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        i : ι
        x✝ : Monoid.PushoutI.NormalWord d
        ⊢ Eq (HSMul.hSMul 1 x✝) x✝
      -/
      dsimp [instHSMul]
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        i : ι
        x✝ : Monoid.PushoutI.NormalWord d
        ⊢ Eq ((Monoid.PushoutI.NormalWord.equivPair i).symm { head := HMul.hMul 1 ((Mo …
      -/
      rw [one_mul]
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        i : ι
        x✝ : Monoid.PushoutI.NormalWord d
        ⊢ Eq ((Monoid.PushoutI.NormalWord.equivPair i).symm { head := ((Monoid.Pushout …
      -/
      exact (equivPair i).symm_apply_apply _
      /-
        🎉 no goals
      -/
    mul_smul := fun _ _ _ => by
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        i : ι
        x✝² x✝¹ : G i
        x✝ : Monoid.PushoutI.NormalWord d
        ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
      -/
      dsimp [instHSMul]
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        i : ι
        x✝² x✝¹ : G i
        x✝ : Monoid.PushoutI.NormalWord d
        ⊢ Eq ((Monoid.PushoutI.NormalWord.equivPair i).symm { head := HMul.hMul (HMul. …
      -/
      simp [mul_assoc, Equiv.apply_symm_apply, Function.End.mul_def] }
      /-
        🎉 no goals
      -/


theorem summand_smul_def' {i : ι} (g : G i) (w : NormalWord d) :
    g • w = (equivPair i).symm
      { equivPair i w with
        head := g * (equivPair i w).head } := rfl


noncomputable instance mulAction : MulAction (PushoutI φ) (NormalWord d) :=
  MulAction.ofEndHom <|
    lift
      (fun _ => MulAction.toEndHom)
      MulAction.toEndHom <| by
    /-
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝⁴ : Monoid K
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      ⊢ ∀ (i : ι), Eq (((fun x => MulAction.toEndHom) i).comp (φ i)) MulAction.toEnd …
    -/
    intro i
    simp only [MulAction.toEndHom, DFunLike.ext_iff, MonoidHom.coe_comp, MonoidHom.coe_mk,
      OneHom.coe_mk, comp_apply]
    /-
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝⁴ : Monoid K
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      i : ι
      ⊢ ∀ (x : H), Eq (fun x2 => HSMul.hSMul ((φ i) x) x2) fun x2 => HSMul.hSMul x x2
    -/
    intro h
    /-
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝⁴ : Monoid K
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      i : ι
      h : H
      ⊢ Eq (fun x2 => HSMul.hSMul ((φ i) h) x2) fun x2 => HSMul.hSMul h x2
    -/
    funext w
    /-
      case h
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝⁴ : Monoid K
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      i : ι
      h : H
      w : Monoid.PushoutI.NormalWord d
      ⊢ Eq (HSMul.hSMul ((φ i) h) w) (HSMul.hSMul h w)
    -/
    apply NormalWord.ext_smul i
    simp only [summand_smul_def', equivPair, rcons, Word.equivPair_symm, Equiv.coe_fn_mk,
      Equiv.coe_fn_symm_mk, Word.equivPair_smul_same, Word.equivPair_tail_eq_inv_smul,
      Word.rcons_eq_smul, equiv_fst_eq_mul_inv, map_mul, map_inv, mul_smul, inv_smul_smul,
      smul_inv_smul, base_smul_def', MonoidHom.apply_ofInjective_symm]


theorem base_smul_def (h : H) (w : NormalWord d) :
    base φ h • w = { w with head := h * w.head } := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    h : H
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (HSMul.hSMul ((Monoid.PushoutI.base φ) h) w) { toWord := w.toWord, head : …
  -/
  dsimp [NormalWord.mulAction, instHSMul, SMul.smul]
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    h : H
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq ((Monoid.PushoutI.lift (fun x => MulAction.toEndHom) MulAction.toEndHom ⋯ …
  -/
  rw [lift_base]
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    h : H
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (MulAction.toEndHom h w) { toWord := w.toWord, head := HMul.hMul h w.head …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem summand_smul_def {i : ι} (g : G i) (w : NormalWord d) :
    of (φ := φ) i g • w = (equivPair i).symm
      { equivPair i w with
        head := g * (equivPair i w).head } := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    g : G i
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (HSMul.hSMul ((Monoid.PushoutI.of i) g) w)
        ((Monoid.PushoutI.NormalWord.equivPair i).symm
          (let __src := (Monoid.PushoutI.NormalWord.equivPair i) w;
          { head := HMul.hMul g ((Monoid.PushoutI.NormalWord.equivPair i) w).head, …
  -/
  dsimp [NormalWord.mulAction, instHSMul, SMul.smul]
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    g : G i
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq ((Monoid.PushoutI.lift (fun x => MulAction.toEndHom) MulAction.toEndHom ⋯ …
  -/
  rw [lift_of]
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    g : G i
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (MulAction.toEndHom g w) ((Monoid.PushoutI.NormalWord.equivPair i).symm { …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem of_smul_eq_smul {i : ι} (g : G i) (w : NormalWord d) :
    of (φ := φ) i g • w = g • w := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    g : G i
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (HSMul.hSMul ((Monoid.PushoutI.of i) g) w) (HSMul.hSMul g w)
  -/
  rw [summand_smul_def, summand_smul_def']
  /-
    🎉 no goals
  -/


theorem base_smul_eq_smul (h : H) (w : NormalWord d) :
    base φ h • w = h • w := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    h : H
    w : Monoid.PushoutI.NormalWord d
    ⊢ Eq (HSMul.hSMul ((Monoid.PushoutI.base φ) h) w) (HSMul.hSMul h w)
  -/
  rw [base_smul_def, base_smul_def']
  /-
    🎉 no goals
  -/


/-- Induction principle for `NormalWord`, that corresponds closely to inducting on
the underlying list. -/
@[elab_as_elim]
noncomputable def consRecOn {motive : NormalWord d → Sort _} (w : NormalWord d)
    (h_empty : motive empty)
    (h_cons : ∀ (i : ι) (g : G i) (w : NormalWord d) (hmw : w.fstIdx ≠ some i)
      (_hgn : g ∈ d.set i) (hgr : g ∉ (φ i).range) (_hw1 : w.head = 1),
      motive w →  motive (cons g w hmw hgr))
    (h_base : ∀ (h : H) (w : NormalWord d), w.head = 1 → motive w → motive
      (base φ h • w)) : motive w := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝⁴ : Monoid K
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    motive : Monoid.PushoutI.NormalWord d → Sort ?u.254962
    w : Monoid.PushoutI.NormalWord d
    h_empty : motive Monoid.PushoutI.NormalWord.empty
    h_cons : (i : ι) → (g : G i) → (w : Monoid.PushoutI.NormalWord d) → (hmw : Ne  …
    h_base : (h : H) → (w : Monoid.PushoutI.NormalWord d) → Eq w.head 1 → motive w …
    ⊢ motive w
  -/
  rcases w with ⟨w, head, h3⟩
  /-
    case mk
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    K : Type u_4
    inst✝⁴ : Monoid K
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    motive : Monoid.PushoutI.NormalWord d → Sort ?u.254962
    h_empty : motive Monoid.PushoutI.NormalWord.empty
    h_cons : (i : ι) → (g : G i) → (w : Monoid.PushoutI.NormalWord d) → (hmw : Ne  …
    h_base : (h : H) → (w : Monoid.PushoutI.NormalWord d) → Eq w.head 1 → motive w …
    w : Monoid.CoprodI.Word G
    head : H
    h3 : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Membership.mem (d.s …
    ⊢ motive { toWord := w, head := head, normalized := h3 }
  -/
  convert h_base head ⟨w, 1, h3⟩ rfl ?_
    /-
      case h.e'_1.h.e'_9
      ι : Type u_1
      G : ι → Type u_2
      H : Type u_3
      K : Type u_4
      inst✝⁴ : Monoid K
      inst✝³ : (i : ι) → Group (G i)
      inst✝² : Group H
      φ : (i : ι) → MonoidHom H (G i)
      d : Monoid.PushoutI.NormalWord.Transversal φ
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (G i)
      motive : Monoid.PushoutI.NormalWord d → Sort ?u.254962
      h_empty : motive Monoid.PushoutI.NormalWord.empty
      h_cons : (i : ι) → (g : G i) → (w : Monoid.PushoutI.NormalWord d) → (hmw : Ne  …
      h_base : (h : H) → (w : Monoid.PushoutI.NormalWord d) → Eq w.head 1 → motive w …
      w : Monoid.CoprodI.Word G
      head : H
      h3 : ∀ (i : ι) (g : G i), Membership.mem w.toList ⟨i, g⟩ → Membership.mem (d.s …
      ⊢ Eq head (HSMul.hSMul ((Monoid.PushoutI.base φ) head) { toWord := w, head :=  …
    -/
  · simp [base_smul_def]
    /-
      🎉 no goals
    -/
  · induction w using Word.consRecOn with
    | h_empty => exact h_empty
    | h_cons i g w h1 hg1 ih =>
      convert h_cons i g ⟨w, 1, fun _ _ h => h3 _ _ (List.mem_cons_of_mem _ h)⟩
        h1 (h3 _ _ (List.mem_cons_self _ _)) ?_ rfl
        (ih ?_)
      · ext
        simp only [Word.cons, Option.mem_def, cons, map_one, mul_one,
          (equiv_snd_eq_self_iff_mem (d.compl i) (one_mem _)).2
          (h3 _ _ (List.mem_cons_self _ _))]
      · apply d.injective i
        simp only [cons, equiv_fst_eq_mul_inv, MonoidHom.apply_ofInjective_symm,
          map_one, mul_one, mul_inv_cancel, (equiv_snd_eq_self_iff_mem (d.compl i) (one_mem _)).2
          (h3 _ _ (List.mem_cons_self _ _))]
      · rwa [← SetLike.mem_coe,
          ← coe_equiv_snd_eq_one_iff_mem (d.compl i) (d.one_mem _),
          (equiv_snd_eq_self_iff_mem (d.compl i) (one_mem _)).2
          (h3 _ _ (List.mem_cons_self _ _))]



theorem cons_eq_smul {i : ι} (g : G i)
    (w : NormalWord d) (hmw : w.fstIdx ≠ some i)
    (hgr : g ∉ (φ i).range) : cons g w hmw hgr = of (φ := φ) i g  • w := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝³ : (i : ι) → Group (G i)
    inst✝² : Group H
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    i : ι
    g : G i
    w : Monoid.PushoutI.NormalWord d
    hmw : Ne w.fstIdx (Option.some i)
    hgr : Not (Membership.mem (φ i).range g)
    ⊢ Eq (Monoid.PushoutI.NormalWord.cons g w hmw hgr) (HSMul.hSMul ((Monoid.Pusho …
  -/
  apply ext_smul i
  simp only [cons, ne_eq, Word.cons_eq_smul, MonoidHom.apply_ofInjective_symm,
    equiv_fst_eq_mul_inv, mul_assoc, map_mul, map_inv, mul_smul, inv_smul_smul, summand_smul_def,
    equivPair, rcons, Word.equivPair_symm, Word.rcons_eq_smul, Equiv.coe_fn_mk,
    Word.equivPair_tail_eq_inv_smul, Equiv.coe_fn_symm_mk, smul_inv_smul]


@[simp]
theorem prod_summand_smul {i : ι} (g : G i) (w : NormalWord d) :
    (g • w).prod = of i g * w.prod := by
  simp only [prod, summand_smul_def', equivPair, rcons, Word.equivPair_symm,
    Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk, Word.equivPair_smul_same,
    Word.equivPair_tail_eq_inv_smul, Word.rcons_eq_smul, ← of_apply_eq_base φ i,
    MonoidHom.apply_ofInjective_symm, equiv_fst_eq_mul_inv, mul_assoc, map_mul, map_inv,
    Word.prod_smul, ofCoprodI_of, inv_mul_cancel_left, mul_inv_cancel_left]


@[simp]
theorem prod_smul (g : PushoutI φ) (w : NormalWord d) :
    (g • w).prod = g * w.prod := by
  induction g using PushoutI.induction_on generalizing w with
  | of i g => rw [of_smul_eq_smul, prod_summand_smul]
  | base h => rw [base_smul_eq_smul, prod_base_smul]
  | mul x y ihx ihy => rw [mul_smul, ihx, ihy, mul_assoc]


theorem prod_smul_empty (w : NormalWord d) : w.prod • empty = w := by
  induction w using consRecOn with
  | h_empty => simp
  | h_cons i g w _ _ _ _ ih =>
    rw [prod_cons, mul_smul, ih, cons_eq_smul]
  | h_base h w _ ih =>
    rw [prod_smul, mul_smul, ih]


/-- The equivalence between normal forms and elements of the pushout -/
noncomputable def equiv : PushoutI φ ≃ NormalWord d :=
  { toFun := fun g => g • .empty
    invFun := fun w => w.prod
    left_inv := fun g => by
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        g : Monoid.PushoutI φ
        ⊢ Eq ((fun w => w.prod) ((fun g => HSMul.hSMul g Monoid.PushoutI.NormalWord.em …
      -/
      simp only [prod_smul, prod_empty, mul_one]
      /-
        🎉 no goals
      -/
    right_inv := fun w => prod_smul_empty w }


theorem prod_injective {ι : Type*} {G : ι → Type*} [(i : ι) → Group (G i)] {φ : (i : ι) → H →* G i}
    {d : Transversal φ} : Function.Injective (prod : NormalWord d → PushoutI φ) := by
  /-
    H : Type u_3
    inst✝¹ : Group H
    ι : Type u_5
    G : ι → Type u_6
    inst✝ : (i : ι) → Group (G i)
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    ⊢ Function.Injective Monoid.PushoutI.NormalWord.prod
  -/
  letI := Classical.decEq ι
  /-
    H : Type u_3
    inst✝¹ : Group H
    ι : Type u_5
    G : ι → Type u_6
    inst✝ : (i : ι) → Group (G i)
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Function.Injective Monoid.PushoutI.NormalWord.prod
  -/
  letI := fun i => Classical.decEq (G i)
  /-
    H : Type u_3
    inst✝¹ : Group H
    ι : Type u_5
    G : ι → Type u_6
    inst✝ : (i : ι) → Group (G i)
    φ : (i : ι) → MonoidHom H (G i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    this✝ : DecidableEq ι := Classical.decEq ι
    this : (i : ι) → DecidableEq (G i) := fun i => Classical.decEq (G i)
    ⊢ Function.Injective Monoid.PushoutI.NormalWord.prod
  -/
  classical exact equiv.symm.injective
  /-
    🎉 no goals
  -/


instance : FaithfulSMul (PushoutI φ) (NormalWord d) :=
               /-
                 ι : Type u_1
                 G : ι → Type u_2
                 H : Type u_3
                 K : Type u_4
                 inst✝⁴ : Monoid K
                 inst✝³ : (i : ι) → Group (G i)
                 inst✝² : Group H
                 φ : (i : ι) → MonoidHom H (G i)
                 d : Monoid.PushoutI.NormalWord.Transversal φ
                 inst✝¹ : DecidableEq ι
                 inst✝ : (i : ι) → DecidableEq (G i)
                 m₁✝ m₂✝ : Monoid.PushoutI φ
                 h : ∀ (a : Monoid.PushoutI.NormalWord d), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul  …
                 ⊢ Eq m₁✝ m₂✝
               -/
  ⟨fun h => by simpa using congr_arg prod (h empty)⟩
               /-
                 🎉 no goals
               -/


instance (i : ι) : FaithfulSMul (G i) (NormalWord d) :=
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        i : ι
        ⊢ ∀ {m₁ m₂ : G i}, (∀ (a : Monoid.PushoutI.NormalWord d), Eq (HSMul.hSMul m₁ a …
      -/
  ⟨by simp [summand_smul_def']⟩
      /-
        🎉 no goals
      -/


instance : FaithfulSMul H (NormalWord d) :=
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        K : Type u_4
        inst✝⁴ : Monoid K
        inst✝³ : (i : ι) → Group (G i)
        inst✝² : Group H
        φ : (i : ι) → MonoidHom H (G i)
        d : Monoid.PushoutI.NormalWord.Transversal φ
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (G i)
        ⊢ ∀ {m₁ m₂ : H}, (∀ (a : Monoid.PushoutI.NormalWord d), Eq (HSMul.hSMul m₁ a)  …
      -/
  ⟨by simp [base_smul_def']⟩
      /-
        🎉 no goals
      -/


/-- All maps into the `PushoutI`, or amalgamated product of groups are injective,
provided all maps in the diagram are injective.

See also `base_injective` -/
theorem of_injective (hφ : ∀ i, Function.Injective (φ i)) (i : ι) :
    Function.Injective (of (φ := φ) i) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    i : ι
    ⊢ Function.Injective ⇑(Monoid.PushoutI.of i)
  -/
  rcases transversal_nonempty φ hφ with ⟨d⟩
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    i : ι
    d : Monoid.PushoutI.NormalWord.Transversal φ
    ⊢ Function.Injective ⇑(Monoid.PushoutI.of i)
  -/
  let _ := Classical.decEq ι
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    i : ι
    d : Monoid.PushoutI.NormalWord.Transversal φ
    x✝ : DecidableEq ι := Classical.decEq ι
    ⊢ Function.Injective ⇑(Monoid.PushoutI.of i)
  -/
  let _ := fun i => Classical.decEq (G i)
  refine Function.Injective.of_comp
    (f := ((· • ·) : PushoutI φ → NormalWord d → NormalWord d)) ?_
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    i : ι
    d : Monoid.PushoutI.NormalWord.Transversal φ
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : (i : ι) → DecidableEq (G i) := fun i => Classical.decEq (G i)
    ⊢ Function.Injective (Function.comp (fun x1 x2 => HSMul.hSMul x1 x2) ⇑(Monoid. …
  -/
  intros _ _ h
  exact eq_of_smul_eq_smul (fun w : NormalWord d =>
    by simp_all [funext_iff, of_smul_eq_smul])


theorem base_injective (hφ : ∀ i, Function.Injective (φ i)) :
    Function.Injective (base φ) := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    ⊢ Function.Injective ⇑(Monoid.PushoutI.base φ)
  -/
  rcases transversal_nonempty φ hφ with ⟨d⟩
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    ⊢ Function.Injective ⇑(Monoid.PushoutI.base φ)
  -/
  let _ := Classical.decEq ι
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    x✝ : DecidableEq ι := Classical.decEq ι
    ⊢ Function.Injective ⇑(Monoid.PushoutI.base φ)
  -/
  let _ := fun i => Classical.decEq (G i)
  refine Function.Injective.of_comp
    (f := ((· • ·) : PushoutI φ → NormalWord d → NormalWord d)) ?_
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    d : Monoid.PushoutI.NormalWord.Transversal φ
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : (i : ι) → DecidableEq (G i) := fun i => Classical.decEq (G i)
    ⊢ Function.Injective (Function.comp (fun x1 x2 => HSMul.hSMul x1 x2) ⇑(Monoid. …
  -/
  intros _ _ h
  exact eq_of_smul_eq_smul (fun w : NormalWord d =>
    by simp_all [funext_iff, base_smul_eq_smul])


/-- A word in `CoprodI` is reduced if none of its letters are in the base group. -/
def Reduced (w : Word G) : Prop :=
  ∀ g, g ∈ w.toList → g.2 ∉ (φ g.1).range


theorem Reduced.exists_normalWord_prod_eq (d : Transversal φ) {w : Word G} (hw : Reduced φ w) :
    ∃ w' : NormalWord d, w'.prod = ofCoprodI w.prod ∧
      w'.toList.map Sigma.fst = w.toList.map Sigma.fst := by
  classical
  induction w using Word.consRecOn with
  | h_empty => exact ⟨empty, by simp, rfl⟩
  | h_cons i g w hIdx hg1 ih =>
    rcases ih (fun _ hg => hw _ (List.mem_cons_of_mem _ hg)) with
      ⟨w', hw'prod, hw'map⟩
    refine ⟨cons g w' ?_ ?_, ?_⟩
    · rwa [Word.fstIdx, ← List.head?_map, hw'map, List.head?_map]
    · exact hw _ (List.mem_cons_self _ _)
    · simp [hw'prod, hw'map]


/-- For any word `w` in the coproduct,
if `w` is reduced (i.e none its letters are in the image of the base monoid), and nonempty, then
`w` itself is not in the image of the base group. -/
theorem Reduced.eq_empty_of_mem_range
    (hφ : ∀ i, Injective (φ i)) {w : Word G} (hw : Reduced φ w)
    (h : ofCoprodI w.prod ∈ (base φ).range) : w = .empty := by
  /-
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    w : Monoid.CoprodI.Word G
    hw : Monoid.PushoutI.Reduced φ w
    h : Membership.mem (Monoid.PushoutI.base φ).range (Monoid.PushoutI.ofCoprodI w …
    ⊢ Eq w Monoid.CoprodI.Word.empty
  -/
  rcases transversal_nonempty φ hφ with ⟨d⟩
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    w : Monoid.CoprodI.Word G
    hw : Monoid.PushoutI.Reduced φ w
    h : Membership.mem (Monoid.PushoutI.base φ).range (Monoid.PushoutI.ofCoprodI w …
    d : Monoid.PushoutI.NormalWord.Transversal φ
    ⊢ Eq w Monoid.CoprodI.Word.empty
  -/
  rcases hw.exists_normalWord_prod_eq d with ⟨w', hw'prod, hw'map⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    w : Monoid.CoprodI.Word G
    hw : Monoid.PushoutI.Reduced φ w
    h : Membership.mem (Monoid.PushoutI.base φ).range (Monoid.PushoutI.ofCoprodI w …
    d : Monoid.PushoutI.NormalWord.Transversal φ
    w' : Monoid.PushoutI.NormalWord d
    hw'prod : Eq w'.prod (Monoid.PushoutI.ofCoprodI w.prod)
    hw'map : Eq (List.map Sigma.fst w'.toList) (List.map Sigma.fst w.toList)
    ⊢ Eq w Monoid.CoprodI.Word.empty
  -/
  rcases h with ⟨h, heq⟩
  have : (NormalWord.prod (d := d) ⟨.empty, h, by simp⟩) = base φ h := by
    simp [NormalWord.prod]
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    w : Monoid.CoprodI.Word G
    hw : Monoid.PushoutI.Reduced φ w
    d : Monoid.PushoutI.NormalWord.Transversal φ
    w' : Monoid.PushoutI.NormalWord d
    hw'prod : Eq w'.prod (Monoid.PushoutI.ofCoprodI w.prod)
    hw'map : Eq (List.map Sigma.fst w'.toList) (List.map Sigma.fst w.toList)
    h : H
    heq : Eq ((Monoid.PushoutI.base φ) h) (Monoid.PushoutI.ofCoprodI w.prod)
    this : Eq { toWord := Monoid.CoprodI.Word.empty, head := h, normalized := ⋯ }. …
    ⊢ Eq w Monoid.CoprodI.Word.empty
  -/
  rw [← hw'prod, ← this] at heq
  suffices w'.toWord = .empty by
    simp [this, @eq_comm _ []] at hw'map
    ext
    simp [hw'map]
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    G : ι → Type u_2
    H : Type u_3
    inst✝¹ : (i : ι) → Group (G i)
    inst✝ : Group H
    φ : (i : ι) → MonoidHom H (G i)
    hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
    w : Monoid.CoprodI.Word G
    hw : Monoid.PushoutI.Reduced φ w
    d : Monoid.PushoutI.NormalWord.Transversal φ
    w' : Monoid.PushoutI.NormalWord d
    hw'prod : Eq w'.prod (Monoid.PushoutI.ofCoprodI w.prod)
    hw'map : Eq (List.map Sigma.fst w'.toList) (List.map Sigma.fst w.toList)
    h : H
    heq : Eq { toWord := Monoid.CoprodI.Word.empty, head := h, normalized := ⋯ }.p …
    this : Eq { toWord := Monoid.CoprodI.Word.empty, head := h, normalized := ⋯ }. …
    ⊢ Eq w'.toWord Monoid.CoprodI.Word.empty
  -/
  rw [← prod_injective heq]
  /-
    🎉 no goals
  -/


/-- The intersection of the images of the maps from any two distinct groups in the diagram
into the amalgamated product is the image of the map from the base group in the diagram. -/
theorem inf_of_range_eq_base_range
    (hφ : ∀ i, Injective (φ i)) {i j : ι} (hij : i ≠ j) :
    (of i).range ⊓ (of j).range = (base φ).range :=
  le_antisymm
    (by
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
        i j : ι
        hij : Ne i j
        ⊢ LE.le (Min.min (Monoid.PushoutI.of i).range (Monoid.PushoutI.of j).range) (M …
      -/
      intro x ⟨⟨g₁, hg₁⟩, ⟨g₂, hg₂⟩⟩
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
        i j : ι
        hij : Ne i j
        x : Monoid.PushoutI φ
        g₁ : G i
        hg₁ : Eq ((Monoid.PushoutI.of i) g₁) x
        g₂ : G j
        hg₂ : Eq ((Monoid.PushoutI.of j) g₂) x
        ⊢ Membership.mem (Monoid.PushoutI.base φ).range x
      -/
      by_contra hx
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
        i j : ι
        hij : Ne i j
        x : Monoid.PushoutI φ
        g₁ : G i
        hg₁ : Eq ((Monoid.PushoutI.of i) g₁) x
        g₂ : G j
        hg₂ : Eq ((Monoid.PushoutI.of j) g₂) x
        hx : Not (Membership.mem (Monoid.PushoutI.base φ).range x)
        ⊢ False
      -/
      have hx1 : x ≠ 1 := by rintro rfl; simp_all only [ne_eq, one_mem, not_true_eq_false]
      have hg₁1 : g₁ ≠ 1 :=
        ne_of_apply_ne (of (φ := φ) i) (by simp_all)
      have hg₂1 : g₂ ≠ 1 :=
        ne_of_apply_ne (of (φ := φ) j) (by simp_all)
      have hg₁r : g₁ ∉ (φ i).range := by
        rintro ⟨y, rfl⟩
        subst hg₁
        exact hx (of_apply_eq_base φ i y ▸ MonoidHom.mem_range.2 ⟨y, rfl⟩)
      have hg₂r : g₂ ∉ (φ j).range := by
        rintro ⟨y, rfl⟩
        subst hg₂
        exact hx (of_apply_eq_base φ j y ▸ MonoidHom.mem_range.2 ⟨y, rfl⟩)
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
        i j : ι
        hij : Ne i j
        x : Monoid.PushoutI φ
        g₁ : G i
        hg₁ : Eq ((Monoid.PushoutI.of i) g₁) x
        g₂ : G j
        hg₂ : Eq ((Monoid.PushoutI.of j) g₂) x
        hx : Not (Membership.mem (Monoid.PushoutI.base φ).range x)
        hx1 : Ne x 1
        hg₁1 : Ne g₁ 1
        hg₂1 : Ne g₂ 1
        hg₁r : Not (Membership.mem (φ i).range g₁)
        hg₂r : Not (Membership.mem (φ j).range g₂)
        ⊢ False
      -/
      let w : Word G := ⟨[⟨_, g₁⟩, ⟨_, g₂⁻¹⟩], by simp_all, by simp_all⟩
      have hw : Reduced φ w := by
        simp only [w, not_exists, ne_eq, Reduced, List.find?, List.mem_cons,
          List.mem_singleton, forall_eq_or_imp, not_false_eq_true, forall_const, forall_eq,
          true_and, hg₁r, hg₂r, List.mem_nil_iff, false_imp_iff, imp_true_iff, and_true,
          inv_mem_iff]
      have := hw.eq_empty_of_mem_range hφ (by
        simp only [w, Word.prod, List.map_cons, List.prod_cons, List.prod_nil,
          List.map_nil, map_mul, ofCoprodI_of, hg₁, hg₂, map_inv, map_one, mul_one,
          mul_inv_cancel, one_mem])
      /-
        ι : Type u_1
        G : ι → Type u_2
        H : Type u_3
        inst✝¹ : (i : ι) → Group (G i)
        inst✝ : Group H
        φ : (i : ι) → MonoidHom H (G i)
        hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
        i j : ι
        hij : Ne i j
        x : Monoid.PushoutI φ
        g₁ : G i
        hg₁ : Eq ((Monoid.PushoutI.of i) g₁) x
        g₂ : G j
        hg₂ : Eq ((Monoid.PushoutI.of j) g₂) x
        hx : Not (Membership.mem (Monoid.PushoutI.base φ).range x)
        hx1 : Ne x 1
        hg₁1 : Ne g₁ 1
        hg₂1 : Ne g₂ 1
        hg₁r : Not (Membership.mem (φ i).range g₁)
        hg₂r : Not (Membership.mem (φ j).range g₂)
        w : Monoid.CoprodI.Word G := { toList := List.cons ⟨i, g₁⟩ (List.cons ⟨j, Inv. …
        hw : Monoid.PushoutI.Reduced φ w
        this : Eq w Monoid.CoprodI.Word.empty
        ⊢ False
      -/
      simp [w, Word.empty] at this)
      /-
        🎉 no goals
      -/
    (le_inf
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝¹ : (i : ι) → Group (G i)
            inst✝ : Group H
            φ : (i : ι) → MonoidHom H (G i)
            hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
            i j : ι
            hij : Ne i j
            ⊢ LE.le (Monoid.PushoutI.base φ).range (Monoid.PushoutI.of i).range
          -/
      (by rw [← of_comp_eq_base i]
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝¹ : (i : ι) → Group (G i)
            inst✝ : Group H
            φ : (i : ι) → MonoidHom H (G i)
            hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
            i j : ι
            hij : Ne i j
            ⊢ LE.le ((Monoid.PushoutI.of i).comp (φ i)).range (Monoid.PushoutI.of i).range
          -/
          rintro _ ⟨h, rfl⟩
          /-
            case intro
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝¹ : (i : ι) → Group (G i)
            inst✝ : Group H
            φ : (i : ι) → MonoidHom H (G i)
            hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
            i j : ι
            hij : Ne i j
            h : H
            ⊢ Membership.mem (Monoid.PushoutI.of i).range (((Monoid.PushoutI.of i).comp (φ …
          -/
          exact MonoidHom.mem_range.2 ⟨φ i h, rfl⟩)
          /-
            🎉 no goals
          -/
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝¹ : (i : ι) → Group (G i)
            inst✝ : Group H
            φ : (i : ι) → MonoidHom H (G i)
            hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
            i j : ι
            hij : Ne i j
            ⊢ LE.le (Monoid.PushoutI.base φ).range (Monoid.PushoutI.of j).range
          -/
      (by rw [← of_comp_eq_base j]
          /-
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝¹ : (i : ι) → Group (G i)
            inst✝ : Group H
            φ : (i : ι) → MonoidHom H (G i)
            hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
            i j : ι
            hij : Ne i j
            ⊢ LE.le ((Monoid.PushoutI.of j).comp (φ j)).range (Monoid.PushoutI.of j).range
          -/
          rintro _ ⟨h, rfl⟩
          /-
            case intro
            ι : Type u_1
            G : ι → Type u_2
            H : Type u_3
            inst✝¹ : (i : ι) → Group (G i)
            inst✝ : Group H
            φ : (i : ι) → MonoidHom H (G i)
            hφ : ∀ (i : ι), Function.Injective ⇑(φ i)
            i j : ι
            hij : Ne i j
            h : H
            ⊢ Membership.mem (Monoid.PushoutI.of j).range (((Monoid.PushoutI.of j).comp (φ …
          -/
          exact MonoidHom.mem_range.2 ⟨φ j h, rfl⟩))
          /-
            🎉 no goals
          -/


